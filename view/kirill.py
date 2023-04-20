import subprocess
import os
import email
import tkinter as tk
from tkinter import scrolledtext
from email.header import decode_header
from tkinter import filedialog
import nmap
import socket
from bs4 import BeautifulSoup
import re


def nmap_scan():
    target_ip = target_ip_entry.get()
    port_range = port_range_entry.get()
    nm = nmap.PortScanner()
    arguments = f'-sS -p {port_range}'
    nm.scan(target_ip, arguments=arguments)
    open_ports = nm[target_ip].all_tcp()

    vulnerabilities_window = tk.Toplevel(root)
    vulnerabilities_window.title('Vulnerabilities')
    vulnerabilities_text = tk.Text(vulnerabilities_window, height=30, width=80)
    vulnerabilities_text.pack()

    for port in open_ports:
        try:
            nm.scan(target_ip, str(port), arguments="-sC -sV -Pn")
            if 'script' in nm[target_ip]['tcp'][port]:
                script_output = nm[target_ip]['tcp'][port]['script']
                vulnerabilities_text.insert(tk.END, f'Port {port}:\n{script_output}\n\n')
        except Exception as e:
            vulnerabilities_text.insert(tk.END, f'Error on port {port}: {str(e)}\n\n')


def analyze_email_headers():
    eml_file = open(email_input_entry.get(), "rb")
    message = email.message_from_binary_file(eml_file)
    header_analysis = ""
    threat_list = []
    for header in message.keys():
        value = message[header]
        if value:
            try:
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='replace')
                else:
                    value = str(value)

                if header.lower() == "from":
                    # Проверяем формат отправителя
                    if not re.match(r"[^@]+@[^@]+\.[^@]+", value):
                        threat_list.append(f'WARNING: suspicious email format in {header}: {value}')

                elif header.lower() in ("to", "cc"):
                    # Проверяем на наличие массовой рассылки
                    if "," in value or len(value.split()) > 1:
                        threat_list.append(f'WARNING: possible mass mailing in {header}: {value}')

                elif header.lower() == "subject":
                    # Проверяем на наличие оскорбительного или угрожающего заголовка
                    if re.search(r"\b(?:offensive|threat|unexpected)\b", value, re.IGNORECASE):
                        threat_list.append(f'WARNING: potentially offensive or threatening subject: {value}')

                elif header.lower() == "reply-to":
                    # Проверяем наличие других подозрительных заголовков
                    if value != message["from"]:
                        threat_list.append(f'WARNING: different reply-to email address: {value}')

                elif header.lower() == "(other suspicious header)":
                    threat_list.append(f'WARNING: suspicious value in {header}: {value}')

                # Проверяем HTML-контент на наличие уязвимости XSS
                if header.lower() == 'content-type' and 'text/html' in value.lower():
                    if check_xss_vulnerabilities(message.as_string()):
                        threat_list.append(f'WARNING: XSS vulnerability found in email content')

                header_analysis += f'{header}: {value}\n'

            except Exception as e:
                header_analysis += f'Error decoding {header}: {str(e)}\n'
                continue

    eml_file.close()

    if threat_list:
        threat_window = tk.Toplevel()
        threat_window.title("Potential Threats Found")
        threat_text = tk.Text(threat_window, height=20, width=80)
        threat_text.pack()
        for threat in threat_list:
            threat_text.insert("end", threat + "\n")
    else:
        tk.messagebox.showinfo("No Threats Found", "No potential threats were found in the email headers.")

    window = tk.Toplevel()
    window.title("Email header analysis")
    window.geometry("800x600")

    output_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=("Courier New", 10))
    output_text.pack(fill=tk.BOTH, expand=True)
    output_text.insert(tk.END, header_analysis)
    output_text.config(state=tk.DISABLED)


def start_tshark_capture():
    global tshark_process
    capture_filename = 'capture.pcapng'
    if os.path.exists(capture_filename):
        os.remove(capture_filename)
    capture_command = ['tshark', '-i', 'Ethernet', '-w', 'capture.pcapng']
    tshark_process = subprocess.Popen(capture_command, stdout=subprocess.PIPE)
    tshark_output.config(state=tk.NORMAL)
    tshark_output.delete('1.0', tk.END)
    tshark_output.insert(tk.END, 'Tshark capture started\n')
    tshark_output.config(state=tk.DISABLED)


def stop_tshark_capture():
    global tshark_process
    if tshark_process:
        tshark_process.terminate()
        tshark_process.wait()
        tshark_output.config(state=tk.NORMAL)
        tshark_output.delete('1.0', tk.END)
        tshark_output.insert(tk.END, 'Tshark capture stopped\n')
        tshark_output.config(state=tk.DISABLED)


def view_pcapng():
    # Команда для просмотра файла в формате pcapng
    view_command = f'tshark -r capture.pcapng'

    # Запускаем процесс и получаем вывод
    view_process = subprocess.Popen(view_command.split(), stdout=subprocess.PIPE)
    view_output, _ = view_process.communicate()

    # Создаем новое окно для отображения вывода
    window = tk.Toplevel()
    window.title("Детали захвата пакетов")
    window.geometry("800x600")

    # Добавляем текстовый виджет и полосу прокрутки в окно
    output_text = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=("Courier New", 10))
    output_text.pack(fill=tk.BOTH, expand=True)

    # Вставляем вывод в текстовый виджет
    output_text.insert(tk.END, view_output.decode('utf-8'))

    # Запрещаем редактирование текстового виджета
    output_text.config(state=tk.DISABLED)


def browse_email_file():
    eml_file = filedialog.askopenfile(mode='rb', title='Choose an email file', filetypes=[('Email files', '*.eml')])
    if eml_file:
        email_input_entry.delete(0, tk.END)
        email_input_entry.insert(0, eml_file.name)


def get_ip(domain):
    try:
        ip = socket.gethostbyname(domain)
        return ip
    except socket.gaierror:
        return None


def check_spf(ip, domain):
    spf_record = domain + '_spf.yandex.ru'
    try:
        spf_response = socket.gethostbyname(spf_record)
        if ip == spf_response:
            return True
        else:
            return False
    except socket.gaierror:
        return None


def check_spf_gui():
    sender_ip = sender_ip_entry.get()
    domain = domain_entry.get()

    domain_ip = get_ip(domain)
    if domain_ip is None:
        result_label.config(text=f'Не удалось получить IP-адрес домена {domain}')
    else:
        is_spf_valid = check_spf(sender_ip, domain)
        if is_spf_valid is None:
            result_label.config(text='Не удалось проверить защиту от спуфинга')
        elif is_spf_valid:
            result_label.config(text=f'IP-адрес отправителя {sender_ip} прошел проверку защиты от спуфинга')
        else:
            result_label.config(text=f'IP-адрес отправителя {sender_ip} не прошел проверку защиты от спуфинга')


def check_xss_vulnerabilities(msg):
    # Разбор сообщения электронной почты
    email_msg = email.message_from_string(msg)

    # Проверка всех частей сообщения на уязвимости XSS
    for part in email_msg.walk():
        # Проверка только тех частей, которые содержат HTML-контент
        if part.get_content_type() == 'text/html':
            # Разбор HTML-контента
            soup = BeautifulSoup(part.get_payload(decode=True), 'html.parser')

            # Проверка всех тегов скриптов на наличие подозрительного содержимого
            for script_tag in soup.find_all('script'):
                script_content = script_tag.string
                if script_content is not None and 'alert' in script_content:
                    return True

            # Проверка всех атрибутов обработчиков событий на наличие подозрительного содержимого
            for tag in soup.find_all():
                for attr in tag.attrs:
                    if attr.startswith('on') and 'alert' in tag.attrs[attr]:
                        return True

    # XSS-уязвимости не обнаружены
    return False


root = tk.Tk()
root.geometry('1050x800')
root.title('Светлая сторона')
root.configure(background='#F5F5F5')

# Define font
font_title = ('Arial', 16, 'bold')
font_label = ('Arial', 12)
font_button = ('Arial', 12, 'bold')

# IP
ip_section = tk.LabelFrame(root, text='NMAP SCANNER', font=font_title, bg='#F5F5F5', fg='#222222', bd=3, padx=10,
                           pady=10)
ip_section.pack(fill='both', expand='yes', padx=10, pady=10)

target_ip_label = tk.Label(ip_section, text='Target IP:', font=font_label, bg='#F5F5F5', fg='#222222')
target_ip_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

target_ip_entry = tk.Entry(ip_section, font=font_label, bg='#FFFFFF', fg='#222222', width=30)
target_ip_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

port_range_label = tk.Label(ip_section, text='Port Range:', font=font_label, bg='#F5F5F5', fg='#222222')
port_range_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')

port_range_entry = tk.Entry(ip_section, font=font_label, bg='#FFFFFF', fg='#222222', width=30)
port_range_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')

scan_button = tk.Button(ip_section, text='Scan', font=font_button, bg='#222222', fg='#FFFFFF',
                        activebackground='#F5F5F5', activeforeground='#222222', command=nmap_scan)
scan_button.grid(row=2, column=1, padx=10, pady=10, sticky='e')

# email
email_section = tk.LabelFrame(root, text='EMAIL HEADERS ANALYZER', font=font_title, bg='#F5F5F5', fg='#222222', bd=3,
                              padx=10, pady=10)
email_section.pack(fill='both', expand='yes', padx=10, pady=10)

email_input_label = tk.Label(email_section, text='Email file:', font=font_label, bg='#F5F5F5', fg='#222222')
email_input_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

email_input_entry = tk.Entry(email_section, font=font_label, bg='#FFFFFF', fg='#222222', width=70)
email_input_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

email_browse_button = tk.Button(email_section, text='Browse', font=font_button, bg='#222222', fg='#FFFFFF',
                                activebackground='#F5F5F5', activeforeground='#222222', command=browse_email_file)
email_browse_button.grid(row=0, column=2, padx=10, pady=10, sticky='e')

analyze_email_button = tk.Button(email_section, text='Analyze email headers', font=font_button, bg='#222222',
                                 fg='#FFFFFF', activebackground='#F5F5F5', activeforeground='#222222',
                                 command=analyze_email_headers)
analyze_email_button.grid(row=1, column=1, padx=10, pady=10, sticky='e')

# SPF
spf_section = tk.LabelFrame(root, text='SPF Checker', font=font_title, bg='#F5F5F5', fg='#222222', bd=3, padx=10,
                            pady=10)
spf_section.pack(fill='both', expand='yes', padx=10, pady=10)

sender_ip_label = tk.Label(spf_section, text='Sender IP:', font=font_label, bg='#F5F5F5', fg='#222222')
sender_ip_label.grid(row=0, column=0, padx=10, pady=10, sticky='w')

sender_ip_entry = tk.Entry(spf_section, font=font_label, bg='#FFFFFF', fg='#222222', width=30)
sender_ip_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

domain_label = tk.Label(spf_section, text='Domain:', font=font_label, bg='#F5F5F5', fg='#222222')
domain_label.grid(row=1, column=0, padx=10, pady=10, sticky='w')

domain_entry = tk.Entry(spf_section, font=font_label, bg='#FFFFFF', fg='#222222', width=30)
domain_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')

check_spf_button = tk.Button(spf_section, text='Check SPF', font=font_button, bg='#222222', fg='#FFFFFF',
                             command=check_spf_gui)
check_spf_button.grid(row=2, column=0, padx=10, pady=10, sticky='w')

result_label = tk.Label(spf_section, text='', font=font_label, bg='#F5F5F5', fg='#222222')
result_label.grid(row=2, column=1, padx=10, pady=10, sticky='w')

tshark_section = tk.LabelFrame(root, text='Tshark Capture', font=font_title, bg='#F5F5F5', fg='#222222', bd=3, padx=10,
                               pady=10)
tshark_section.pack(fill='both', expand='yes', padx=10, pady=10)

start_button = tk.Button(tshark_section, text='Start Capture', font=font_button, bg='#FFFFFF', fg='#222222',
                         command=start_tshark_capture)
start_button.pack(side='left', padx=10)

stop_button = tk.Button(tshark_section, text='Stop Capture', font=font_button, bg='#FFFFFF', fg='#222222',
                        command=stop_tshark_capture)
stop_button.pack(side='left', padx=10)

view_button = tk.Button(tshark_section, text='View .pcapng', font=font_button, bg='#FFFFFF', fg='#222222',
                        command=view_pcapng)
view_button.pack(side='left', padx=10)

tshark_output = tk.Text(tshark_section, font=font_label, bg='#FFFFFF', fg='#222222', width=50, height=10)
tshark_output.pack(padx=10, pady=10)

root.mainloop()