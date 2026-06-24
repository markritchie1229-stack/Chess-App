#!/usr/bin/env python3
"""Annotate chess puzzles with a truncated winning line.

This version is hard-coded to read Skewers.Complete.json and write
Skewers.Complete.Annotated.json.

Rule used to keep counting the sequence:
- Only odd plies (the solver's moves) are recorded.
- On odd plies, ask Stockfish for the best and second-best moves.
- Keep the solver move only if the best move is at least MIN_EDGE_CP
  centipawns better than the second-best move.
- Even plies (the opponent's replies) are played internally only.
- Even plies are NOT written to the output.
- Stop the sequence as soon as a solver move fails the threshold.

The output preserves the original puzzle fields and adds:
- winning_sequence: list of solver plies only (odd plies)
- counted_plies: number of solver moves kept
- final_sequence_eval_cp: evaluation after the last kept solver ply
- sequence_cutoff_reason: why the line stopped
- first_move_source: whether the first move came from the provided solution
  or from the engine

Requirements:
    pip install python-chess
    Stockfish installed and available on PATH.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import chess.engine

INPUT_FILE = Path("all_forks.filtered.shuffled.json")
OUTPUT_FILE = Path("Forks.Complete.Annotated.json")

ENGINE_PATH = "stockfish"
DEPTH = 12
MIN_EDGE_CP = 150
MAX_SOLVER_MOVES = 12
REQUIRE_SEQUENCE_LENGTH = 0

MATE_SCORE = 100000


@dataclass
class SequenceStep:
    ply: int
    uci: str
    san: str
    eval_cp: int
    delta_cp: int
    edge_cp: int
    best_move_cp: int
    second_move_cp: int
    side_to_move: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ply": self.ply,
            "uci": self.uci,
            "san": self.san,
            "eval_cp": self.eval_cp,
            "delta_cp": self.delta_cp,
            "edge_cp": self.edge_cp,
            "best_move_cp": self.best_move_cp,
            "second_move_cp": self.second_move_cp,
            "side_to_move": self.side_to_move,
            "source": self.source,
        }


def score_to_cp(pov_score: chess.engine.PovScore) -> int:
    if pov_score.is_mate():
        mate = pov_score.mate()
        if mate is None:
            return 0
        return MATE_SCORE if mate > 0 else -MATE_SCORE

    cp = pov_score.score(mate_score=MATE_SCORE)
    return cp if cp is not None else 0


def eval_for_color(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    color: bool,
    depth: int,
) -> int:
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return score_to_cp(info["score"].pov(color))


def analyze_solver_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
) -> Tuple[chess.Move, int, int, int]:
    """Return best move, best score, second score, and edge for a solver turn."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=2)
    if not isinstance(info, list):
        info = [info]

    lines: List[Tuple[chess.Move, int]] = []
    for item in info[:2]:
        pv = item.get("pv") or []
        if not pv:
            continue
        move = pv[0]
        score_cp = score_to_cp(item["score"].pov(board.turn))
        lines.append((move, score_cp))

    if len(lines) < 2:
        raise ValueError("Engine returned fewer than 2 lines for a solver turn")

    best_move, best_score = lines[0]
    second_score = lines[1][1]
    edge_cp = best_score - second_score
    return best_move, best_score, second_score, edge_cp


def analyze_opponent_position(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    depth: int,
) -> Tuple[chess.Move, int]:
    """Return best move and best score for an opponent turn, with no edge analysis."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    pv = info.get("pv") or []
    if pv:
        best_move = pv[0]
    else:
        best_move = next(iter(board.legal_moves))
    best_score = score_to_cp(info["score"].pov(board.turn))
    return best_move, best_score


def choose_first_move(
    board: chess.Board,
    solution: Optional[str],
    fallback_move: chess.Move,
) -> Tuple[chess.Move, str]:
    """Use the provided solution on the first solver ply if legal; otherwise use engine best."""
    if solution:
        try:
            move = chess.Move.from_uci(solution)
            if board.is_legal(move):
                return move, "solution"
        except ValueError:
            pass
    return fallback_move, "engine"


def build_sequence(
    engine: chess.engine.SimpleEngine,
    puzzle: Dict[str, Any],
    depth: int,
    min_edge_cp: int,
    max_solver_moves: int,
) -> Dict[str, Any]:
    board = chess.Board(puzzle["fen"])
    root_color = board.turn

    prev_eval = eval_for_color(engine, board, root_color, depth)

    winning_sequence: List[SequenceStep] = []
    cutoff_reason = "max_solver_moves"
    first_move_source: Optional[str] = None

    solver_move_index = 0

    while solver_move_index < max_solver_moves and not board.is_game_over(claim_draw=True):
        # Solver ply (odd ply from the perspective of the sequence).
        best_move, best_score, second_score, edge_cp = analyze_solver_position(
            engine=engine,
            board=board,
            depth=depth,
        )

        if edge_cp < min_edge_cp:
            cutoff_reason = f"solver_edge_below_{min_edge_cp}"
            break

        if solver_move_index == 0:
            move, first_move_source = choose_first_move(
                board=board,
                solution=puzzle.get("solution"),
                fallback_move=best_move,
            )
        else:
            move = best_move

        if not board.is_legal(move):
            cutoff_reason = "illegal_solver_move"
            break

        odd_ply_number = solver_move_index * 2 + 1
        san = board.san(move)
        side_to_move = "white" if board.turn == chess.WHITE else "black"

        board.push(move)
        current_eval = eval_for_color(engine, board, root_color, depth)
        delta = current_eval - prev_eval

        winning_sequence.append(
            SequenceStep(
                ply=odd_ply_number,
                uci=move.uci(),
                san=san,
                eval_cp=current_eval,
                delta_cp=delta,
                edge_cp=edge_cp,
                best_move_cp=best_score,
                second_move_cp=second_score,
                side_to_move=side_to_move,
                source="solution" if solver_move_index == 0 and first_move_source == "solution" else "engine",
            )
        )
        prev_eval = current_eval
        solver_move_index += 1

        if board.is_game_over(claim_draw=True):
            cutoff_reason = "game_over"
            break

        # Opponent reply is played internally only.
        opp_info = engine.analyse(board, chess.engine.Limit(depth=depth))
        opp_pv = opp_info.get("pv") or []
        if opp_pv:
            opp_move = opp_pv[0]
        else:
            opp_move = next(iter(board.legal_moves))

        if not board.is_legal(opp_move):
            cutoff_reason = "illegal_opponent_move"
            break

        board.push(opp_move)

        if board.is_game_over(claim_draw=True):
            cutoff_reason = "game_over"
            break

    annotated = dict(puzzle)
    annotated["winning_sequence"] = [step.to_dict() for step in winning_sequence]
    annotated["counted_plies"] = len(winning_sequence)
    annotated["final_sequence_eval_cp"] = winning_sequence[-1].eval_cp if winning_sequence else prev_eval
    annotated["sequence_cutoff_reason"] = cutoff_reason
    annotated["first_move_source"] = first_move_source
    return annotated


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE.resolve()}")

    data = json.loads(INPUT_FILE.read_text())
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of puzzle objects")

    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    annotated: List[Dict[str, Any]] = []

    try:
        for idx, puzzle in enumerate(data, start=1):
            if not isinstance(puzzle, dict) or "fen" not in puzzle:
                continue

            try:
                result = build_sequence(
                    engine=engine,
                    puzzle=puzzle,
                    depth=DEPTH,
                    min_edge_cp=MIN_EDGE_CP,
                    max_solver_moves=MAX_SOLVER_MOVES,
                )
            except Exception as exc:
                result = dict(puzzle)
                result["winning_sequence"] = []
                result["counted_plies"] = 0
                result["final_sequence_eval_cp"] = None
                result["sequence_cutoff_reason"] = f"error: {exc}"
                result["first_move_source"] = None

            if REQUIRE_SEQUENCE_LENGTH and result.get("counted_plies", 0) < REQUIRE_SEQUENCE_LENGTH:
                continue

            annotated.append(result)

            if idx % 100 == 0:
                print(f"Processed {idx}/{len(data)} puzzles")

    finally:
        engine.quit()

    OUTPUT_FILE.write_text(json.dumps(annotated, indent=2))
    print(f"Wrote {len(annotated)} puzzles to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()