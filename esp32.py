# pc_bt_spp_duplex.py
import serial
import threading
import sys
import time

# ==== ตั้งค่าพอร์ต ====
PORT = "COM10"  # Windows เช่น "COM7"
# PORT = "/dev/cu.ESP32-SPP-xxxx"   # macOS
# PORT = "/dev/rfcomm0"             # Linux (หลัง bind ด้วย rfcomm)
BAUD = 115200  # สำหรับ SPP ใส่เป็นพิธี (ค่าจริงไม่กระทบ)
TIMEOUT_S = 1


def reader(ser: serial.Serial, stop_flag):
    """เธรดอ่านต่อเนื่อง: พิมพ์ทุกบรรทัดที่ ESP32 ส่งมา (รวม heartbeat)"""
    buf = b""
    while not stop_flag["stop"]:
        try:
            line = ser.readline()  # รอจนหมดบรรทัดหรือ timeout
            if line:
                try:
                    text = line.decode("utf-8", errors="ignore").rstrip("\r\n")
                except:
                    text = str(line)
                print(f"\rESP32> {text}\n> ", end="", flush=True)
            else:
                # ไม่มีข้อมูล ก็วนต่อไป (ปล่อยให้ loop เบาๆ)
                pass
        except serial.SerialException as e:
            print(f"\n[ERR] Serial exception: {e}")
            break
        except Exception as e:
            print(f"\n[ERR] {e}")
            time.sleep(0.1)


def main():
    print(f"Connecting to {PORT} ...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT_S)
    except Exception as e:
        print(f"[ERR] เปิดพอร์ตไม่สำเร็จ: {e}")
        sys.exit(1)

    stop_flag = {"stop": False}
    t = threading.Thread(target=reader, args=(ser, stop_flag), daemon=True)
    t.start()

    print("Connected. พิมพ์คำสั่งแล้วกด Enter (เช่น: PING, ON, OFF, ADC).")
    print("กด Ctrl+C เพื่อออก")
    try:
        while True:
            # prompt ผู้ใช้
            cmd = input("> ").strip()
            if not cmd:
                continue
            # ส่งพร้อม newline ให้เข้ากับโค้ด ESP32 ที่ใช้ readStringUntil('\n')
            ser.write((cmd + "\n").encode("utf-8"))
            ser.flush()
            # ไม่ต้อง read ตรงนี้ ปล่อยให้เธรด reader แสดงผลทุกอย่างที่ตอบกลับมา
    except KeyboardInterrupt:
        print("\nกำลังออก...")
    finally:
        stop_flag["stop"] = True
        try:
            ser.close()
        except:
            pass
        time.sleep(0.2)


if __name__ == "__main__":
    main()
