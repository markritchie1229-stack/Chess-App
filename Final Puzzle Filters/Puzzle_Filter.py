#!/usr/bin/env python3
import json
from pathlib import Path

import chess
import chess.engine

INPUT_FILE = Path("skewers_merged_shuffled.json")
OUTPUT_FILE = Path("Skewers.Complete.json")
STOCKFISH_PATH = "stockfish"

ENGINE_DEPTH = 8
ROOT_MULTIPV = 2
SECOND_BEST_MAX_CP = 3

MATERIAL_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

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

def score_to_cp(pov_score: chess.engine.PovScore) -> int:
    if pov_score.is_mate():
        mate = pov_score.mate()
        if mate is None:
            return 0
        return 100000 if mate > 0 else -100000
    cp = pov_score.score(mate_score=100000)
    return cp if cp is not None else 0

def second_best_root_score(engine: chess.engine.SimpleEngine, board: chess.Board, color: bool) -> int | None:
    info = engine.analyse(
        board,
        chess.engine.Limit(depth=ENGINE_DEPTH),
        multipv=ROOT_MULTIPV,
    )
    if not isinstance(info, list):
        info = [info]

    scores = []
    for line in info:
        pv = line.get("pv") or []
        if not pv:
            continue
        scores.append(score_to_cp(line["score"].pov(color)))

    if len(scores) < 2:
        return None

    scores.sort(reverse=True)
    return scores[1]

def main() -> None:
    data = json.loads(INPUT_FILE.read_text())
    kept = []
    removed = []

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    try:
        for item in data:
            fen = item["fen"]
            solution = item["solution"]

            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(solution)
                if not board.is_legal(move):
                    removed.append((item, "illegal solution"))
                    continue

                mover_color = board.turn

                # Remove positions where the second-best starting move is
                # already better than +3 cp for the player to move.
                second_best = second_best_root_score(engine, board, mover_color)
                if second_best is None:
                    removed.append((item, "no root multipv"))
                    continue
                if second_best > SECOND_BEST_MAX_CP:
                    removed.append(
                        (item, f"second-best root move too good: {second_best} cp")
                    )
                    continue

                start_material = material_balance(board, mover_color)

                # Apply the puzzle move.
                board.push(move)

                # Opponent best reply.
                reply_info = engine.analyse(
                    board,
                    chess.engine.Limit(depth=ENGINE_DEPTH),
                )
                reply_pv = reply_info.get("pv") or []
                if not reply_pv:
                    removed.append((item, "no opponent pv"))
                    continue

                reply = reply_pv[0]
                if not board.is_legal(reply):
                    removed.append((item, "illegal opponent reply"))
                    continue

                board.push(reply)

                # Our best follow-up.
                followup_info = engine.analyse(
                    board,
                    chess.engine.Limit(depth=ENGINE_DEPTH),
                )
                followup_pv = followup_info.get("pv") or []
                if not followup_pv:
                    removed.append((item, "no follow-up pv"))
                    continue

                followup = followup_pv[0]
                if not board.is_legal(followup):
                    removed.append((item, "illegal follow-up"))
                    continue

                board.push(followup)

                # Opponent's next reply.
                final_info = engine.analyse(
                    board,
                    chess.engine.Limit(depth=ENGINE_DEPTH),
                )
                final_pv = final_info.get("pv") or []
                if not final_pv:
                    removed.append((item, "no final pv"))
                    continue

                final_reply = final_pv[0]
                if not board.is_legal(final_reply):
                    removed.append((item, "illegal final reply"))
                    continue

                board.push(final_reply)

                final_material = material_balance(board, mover_color)

                if final_material > start_material:
                    kept.append(item)
                else:
                    removed.append((item, "no material gain after 4 plies"))

            except Exception as exc:
                removed.append((item, f"error: {exc}"))

    finally:
        engine.quit()

    OUTPUT_FILE.write_text(json.dumps(kept, indent=2))
    print(f"Kept {len(kept)} of {len(data)} puzzles")
    print(f"Wrote: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
