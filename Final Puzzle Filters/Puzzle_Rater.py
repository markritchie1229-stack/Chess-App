#!/usr/bin/env python3
import json
from pathlib import Path

import chess
import chess.engine

INPUT_FILE = Path("Forks.Complete.Annotated.json")
OUTPUT_FILE = Path("Forks.Complete.Rated.json")

STOCKFISH_PATH = "stockfish"
DEPTH = 10
MULTIPV = 5

# Change this to 1000 if you want the scale tighter.
MAX_DIFFICULTY = 1500

MATE_SCORE = 100000

MATERIAL_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

# Sequence bonus settings.
# Longer winning sequences increase the final rating, up to the cap.
SEQUENCE_BONUS_PER_SOLVER_MOVE = 35
SEQUENCE_BONUS_CAP = 180


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def score_to_cp(score: chess.engine.PovScore) -> int:
    if score.is_mate():
        mate = score.mate()
        if mate is None:
            return 0
        return MATE_SCORE if mate > 0 else -MATE_SCORE

    cp = score.score(mate_score=MATE_SCORE)
    return cp if cp is not None else 0


def material_balance(board: chess.Board, color: bool) -> int:
    own = 0
    enemy = 0
    for piece in board.piece_map().values():
        if piece.piece_type == chess.KING:
            continue
        value = MATERIAL_VALUES.get(piece.piece_type, 0)
        if piece.color == color:
            own += value
        else:
            enemy += value
    return own - enemy


def fork_targets(board: chess.Board, move: chess.Move):
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN:
        return []

    mover_color = piece.color
    enemy = not mover_color

    board.push(move)
    try:
        targets = []
        for sq in board.attacks(move.to_square):
            p = board.piece_at(sq)
            if p and p.color == enemy and p.piece_type in (
                chess.KNIGHT,
                chess.BISHOP,
                chess.ROOK,
                chess.QUEEN,
            ):
                targets.append(p)
        return targets
    finally:
        board.pop()


def sequence_length_from_item(item):
    if isinstance(item.get("solver_moves_kept"), int):
        return item["solver_moves_kept"]

    seq = item.get("winning_sequence") or []
    if isinstance(seq, list):
        return sum(1 for step in seq if isinstance(step, dict) and step.get("counts_for_threshold", True))
    return 0


def sequence_bonus(item):
    solver_moves = sequence_length_from_item(item)
    bonus = solver_moves * SEQUENCE_BONUS_PER_SOLVER_MOVE
    return min(SEQUENCE_BONUS_CAP, bonus)


def puzzle_difficulty(engine: chess.engine.SimpleEngine, item):
    fen = item["fen"]
    solution = item["solution"]

    board = chess.Board(fen)
    move = chess.Move.from_uci(solution)

    if not board.is_legal(move):
        return None

    mover = board.turn

    # Root eval with multipv so we can compare the chosen move to alternatives.
    root_info = engine.analyse(
        board,
        chess.engine.Limit(depth=DEPTH),
        multipv=MULTIPV,
    )
    if not isinstance(root_info, list):
        root_info = [root_info]

    root_scores = []
    solution_score = None

    for line in root_info:
        pv = line.get("pv") or []
        if not pv:
            continue

        first_move = pv[0]
        cp = score_to_cp(line["score"].pov(mover))
        root_scores.append((first_move, cp))

        if first_move == move:
            solution_score = cp

    if solution_score is None:
        return None

    root_scores.sort(key=lambda x: x[1], reverse=True)
    if len(root_scores) >= 2:
        second_best_score = root_scores[1][1]
    else:
        second_best_score = solution_score

    edge = solution_score - second_best_score

    # Basic tactical shape.
    targets = fork_targets(board, move)
    target_count = len(targets)

    board.push(move)
    try:
        # Best reply from the opponent.
        reply_info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
        reply_pv = reply_info.get("pv") or []
        reply_count = len(reply_pv)

        # A few quick features that usually correlate with difficulty.
        is_check = board.is_check()
        is_capture = board.is_capture(move)

        target_material = 0
        for p in targets:
            target_material += MATERIAL_VALUES.get(p.piece_type, 0)

        start_material = material_balance(chess.Board(fen), mover)
        after_move_material = material_balance(board, mover)
        material_gain = after_move_material - start_material

    except Exception:
        return None
    finally:
        board.pop()

    # Start in the middle of the range.
    difficulty = 750.0

    # Longer winning sequences should rate harder.
    difficulty += sequence_bonus(item)

    # Smaller engine edge => harder.
    if edge <= 10:
        difficulty += 350
    elif edge <= 25:
        difficulty += 250
    elif edge <= 50:
        difficulty += 180
    elif edge <= 100:
        difficulty += 100
    elif edge >= 300:
        difficulty -= 150
    elif edge >= 150:
        difficulty -= 75

    # More equal positions are harder.
    if abs(solution_score) <= 25:
        difficulty += 120
    elif abs(solution_score) <= 50:
        difficulty += 80
    elif abs(solution_score) <= 100:
        difficulty += 40
    elif abs(solution_score) >= 500:
        difficulty -= 120

    # Quiet moves are harder; forcing moves are easier.
    if is_check:
        difficulty -= 140
    if is_capture:
        difficulty -= 70

    # More fork targets usually means easier.
    if target_count >= 3:
        difficulty -= 80
    elif target_count == 2:
        difficulty -= 30

    # Winning more material makes it easier.
    if target_material >= 14:
        difficulty -= 90
    elif target_material >= 9:
        difficulty -= 50

    # If there are many strong candidate replies, that can make it harder.
    if reply_count >= 3:
        difficulty += 40

    # If the move already wins material immediately, it is easier.
    if material_gain > 0:
        difficulty -= min(120, material_gain * 20)

    difficulty = round(clamp(difficulty, 0, MAX_DIFFICULTY))

    return {
        "difficulty": difficulty,
        "edge_cp": edge,
        "solution_eval": solution_score,
        "second_best_eval": second_best_score,
        "target_count": target_count,
        "is_check": is_check,
        "is_capture": is_capture,
        "solver_moves_kept": sequence_length_from_item(item),
        "sequence_bonus": sequence_bonus(item),
    }


def main():
    data = json.loads(INPUT_FILE.read_text())
    rated = []

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    try:
        for i, item in enumerate(data, 1):
            try:
                result = puzzle_difficulty(engine, item)
                if result is None:
                    continue

                item.update(result)
                rated.append(item)

                if i % 100 == 0:
                    print(f"Processed {i} puzzles")

            except Exception as exc:
                print("error:", exc)

    finally:
        engine.quit()

    OUTPUT_FILE.write_text(json.dumps(rated, indent=2))
    print(f"Rated: {len(rated)}")
    print(f"Wrote: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()