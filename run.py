import subprocess
import sys
# 要依次运行的脚本文件列表
scripts = ["BERT.py","mental-roberta.py","BERT_LSTM.py"]

# 打开一个文件，准备将输出写入其中
with open("./results/BERTLSTM_mental_output.txt", "w") as output_file:
    # 循环遍历列表，依次运行每个脚本
    for script in scripts:
        # 使用 Popen 来运行脚本，并将标准输出和标准错误输出重定向到文件和控制台中
        process = subprocess.Popen(["python", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   universal_newlines=True)
        sys.stdout.reconfigure(encoding='utf-8')
        # 读取标准输出和标准错误输出，并同时输出到控制台和文件中
        for line in process.stdout:
            print(line, end='')  # 在控制台输出
            output_file.write(line)  # 写入文件中

        for line in process.stderr:
            print(line, end='')  # 在控制台输出
            output_file.write(line)  # 写入文件中

        # 等待进程结束
        process.communicate()
print("Scripts executed and output saved to output.txt")