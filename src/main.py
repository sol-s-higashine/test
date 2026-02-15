from __future__ import annotations

import operator


OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def calculate_expression(expression: str) -> float:
    tokens = expression.strip().split()
    if len(tokens) != 3:
        raise ValueError("式は '数値 演算子 数値' 形式で入力してください。")

    left_raw, op, right_raw = tokens
    if op not in OPERATORS:
        raise ValueError("利用できる演算子は +, -, *, / のみです。")

    left = float(left_raw)
    right = float(right_raw)
    if op == "/" and right == 0:
        raise ZeroDivisionError("0 で割ることはできません。")

    return OPERATORS[op](left, right)


def calculator() -> None:
    print("=== 電卓プログラム ===")
    print("使い方: 例）2 + 3  または 10 / 5")

    while True:
        expression = input("式を入力してください（終了: exit）> ").strip()

        if expression.lower() == "exit":
            print("終了します。")
            break

        try:
            result = calculate_expression(expression)
            if result.is_integer():
                print("結果:", int(result))
            else:
                print("結果:", result)
        except Exception as exc:
            print(f"エラー: {exc}")


if __name__ == "__main__":
    calculator()
