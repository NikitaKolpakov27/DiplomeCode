import imaplib
import email
import conf_utils


def save_letter(data, path):
    with open(path, "w") as file:
        file.write(data)

def get_sent_emails():
    mail = imaplib.IMAP4_SSL('imap.mail.ru')

    mail_login = input("Введите адрес ящика: ")
    mail_passwd = input("Введите пароль от внешнего ящика: ")

    mail.login(mail_login, mail_passwd)

    mail.list()
    print(mail.list())

    mail.select("&BB4EQgQ,BEAEMAQyBDsENQQ9BD0ESwQ1-", readonly=True) #Выбранное имя ящика берется из mail.list()

    result, data = mail.search(None, "ALL")

    # получение информации о самом "свежем" письме ( в зашифрованном виде )
    ids = data[0]
    id_list = ids.split()

    print("Processing", end="")
    for i in range(len(id_list) - 1, 0, -1):
        print(".", end="")
        letter_data = ""
        latest_email_id = id_list[i]

        result, data = mail.fetch(latest_email_id, "(RFC822)")
        raw_email = data[0][1]
        raw_email_string = raw_email.decode('utf-8')

    # Получение заголовков письма
        email_message = email.message_from_string(raw_email_string)
        letter_data += "=========ЗАГОЛОВКИ ПИСЬМА==========\n"
        letter_data += "FROM: " + str(email.utils.parseaddr(email_message['From'])) + "\n"
        letter_data += "TO: " + str(email_message['To']) + "\n"
        letter_data += "DATE: " + str(email_message['Date']) + "\n"
        letter_data += "SUBJECT: " + str(email_message['Subject']) + "\n"
        letter_data += "MESSAGE_ID: " + str(email_message['Message-Id']) + "\n"
        letter_data += "\n"

    # Получение тела письма (содержимое)
        email_message = email.message_from_string(raw_email_string)
        letter_data_body = "\n=====ТЕЛО ПИСЬМА==========="

        if email_message.is_multipart():

            for payload in email_message.get_payload():

                try:
                    body = payload.get_payload(decode=True).decode('utf-8')
                    letter_data_body += str(body)

                    if tools.is_file_or_text_confidential(True, letter_data_body):
                        path_to_save = "./sent_letters/" + "[" + str(len(id_list) - i) + "]" + "[CONFIDENTIAL]" + ".txt"
                    else:
                        path_to_save = "./sent_letters/" + "[" + str(len(id_list) - i) + "]" + ".txt"

                    letter_data += letter_data_body
                    save_letter(letter_data, path_to_save)
                except Exception as e:
                    letter_data_body += str(e)
                    path_to_save = "./sent_letters/" + "[" + str(len(id_list) - i) + "]" + ".txt"

                    letter_data += letter_data_body
                    save_letter(letter_data, path_to_save)
        else:
            body = email_message.get_payload(decode=True).decode('utf-8')
            letter_data_body += str(body)

            if tools.is_file_or_text_confidential(True, letter_data_body):
                path_to_save = "./sent_letters/" + "[" + str(len(id_list) - i) + "]" + "[CONFIDENTIAL]" + ".txt"
            else:
                path_to_save = "./sent_letters/" + "[" + str(len(id_list) - i) + "]" + ".txt"

            letter_data += letter_data_body
            save_letter(letter_data, path_to_save)


if __name__ == "__main__":
    get_sent_emails()

