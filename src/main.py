def calculator():
    print("=== 電卓プログラム ===")
    print("使い方: 例）2 + 3  または 10 / 5")

    while True:
        expression = input("式を入力してください（終了: exit）> ")

        if expression.lower() == "exit":
            print("終了します。")
            break

        try:
            # eval を使うと式をそのまま計算できる
            result = eval(expression)
            print("結果:", result)
        except Exception as e:
            print("エラーがあります。正しく式を入力してください。")

calculator()

