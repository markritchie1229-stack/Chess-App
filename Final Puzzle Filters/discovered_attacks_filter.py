import json
from pathlib import Path

import chess

INPUT_FILE = Path("discovered_attacks.json")
OUTPUT_FILE = Path("discovered_attacks.filtered.json")


def square_to_coords(square: str):
    """
    Convert algebraic square like 'e4' to (file, rank) where:
      a1 -> (0, 0), h8 -> (7, 7)
    """
    if not square or len(square) != 2:
        raise ValueError(f"Bad square: {square!r}")
    file_char, rank_char = square[0].lower(), square[1]
    if file_char < "a" or file_char > "h" or rank_char < "1" or rank_char > "8":
        raise ValueError(f"Bad square: {square!r}")
    return ord(file_char) - ord("a"), int(rank_char) - 1


def same_diagonal(a: str, b: str) -> bool:
    af, ar = square_to_coords(a)
    bf, br = square_to_coords(b)
    return abs(af - bf) == abs(ar - br)


def same_file_or_rank(a: str, b: str) -> bool:
    af, ar = square_to_coords(a)
    bf, br = square_to_coords(b)
    return af == bf or ar == br


def piece_type(piece_symbol: str) -> str:
    """
    Returns lower-case piece type letter:
      'B', 'b' -> 'b'
      'R', 'r' -> 'r'
      'Q', 'q' -> 'q'
      'P', 'p' -> 'p'
    """
    if not piece_symbol:
        return ""
    return piece_symbol.lower()


def is_pawn_capture(entry: dict) -> bool:
    """
    Exclude cases where the blocker is a pawn and the saved solution is a capture.
    This removes "file opening" discovered attacks that are just pawn captures.
    """
    blocker = entry.get("blocker", {})
    blocker_piece = piece_type(blocker.get("piece", ""))

    if blocker_piece != "p":
        return False

    fen = entry.get("fen", "")
    solution = entry.get("solution", "")

    if not fen or not solution:
        return False

    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(solution)
    except Exception:
        return False

    return board.is_capture(move)


def should_exclude(entry: dict) -> bool:
    slider = entry.get("slider", {})
    blocker = entry.get("blocker", {})

    slider_piece = piece_type(slider.get("piece", ""))
    blocker_piece = piece_type(blocker.get("piece", ""))
    slider_sq = slider.get("square", "")
    blocker_sq = blocker.get("square", "")

    if not slider_piece or not blocker_piece or not slider_sq or not blocker_sq:
        return False

    # Exclude pawn captures that simply open a file.
    if is_pawn_capture(entry):
        return True

    pair = frozenset((slider_piece, blocker_piece))

    # 1. Bishop and Bishop diagonal to each other
    if pair == frozenset(("b", "b")):
        return same_diagonal(slider_sq, blocker_sq)

    # 2. Rook and Rook horizontal or vertical to each other
    if pair == frozenset(("r", "r")):
        return same_file_or_rank(slider_sq, blocker_sq)

    # 3. Queen and bishop diagonal, or queen and rook horizontal/vertical
    if pair == frozenset(("q", "b")):
        return same_diagonal(slider_sq, blocker_sq)

    if pair == frozenset(("q", "r")):
        return same_file_or_rank(slider_sq, blocker_sq)

    return False


def main():
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("Expected the input JSON to be a list of puzzle objects.")

    kept = []
    removed_by_filter = 0
    removed_by_fen = 0
    seen_fens = set()

    for entry in data:
        if should_exclude(entry):
            removed_by_filter += 1
            continue

        fen = entry.get("fen", "")
        if not fen:
            # Keep malformed entries rather than silently dropping them.
            kept.append(entry)
            continue

        if fen in seen_fens:
            removed_by_fen += 1
            continue

        seen_fens.add(fen)
        kept.append(entry)

    OUTPUT_FILE.write_text(json.dumps(kept, indent=2), encoding="utf-8")

    print(f"Loaded:          {len(data)}")
    print(f"Removed filters:  {removed_by_filter}")
    print(f"Removed duplicates: {removed_by_fen}")
    print(f"Kept:             {len(kept)}")
    print(f"Wrote:            {OUTPUT_FILE}")


if __name__ == "__main__":
    main()