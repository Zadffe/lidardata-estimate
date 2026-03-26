# seconds_to_hms.py

def seconds_to_hms(seconds: int):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return hours, minutes, secs

if __name__ == "__main__":
    s = int(input("请输入秒数(整数): ").strip())

    h, m, sec = seconds_to_hms(s)

    print(f"{s} 秒 = {h} 小时 {m} 分 {sec} 秒")
    print(f"{s} 秒 = {s/3600:.6f} 小时（小数形式）")
