import random
import telebot

bot = telebot.TeleBot('5275805360:AAGJb_iy7lyvxlHBNjwKKyLp85Ugm2L9fes')
curr_passwd = ''

def create_password():
    chars = 'abcdefghijklnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
    password = ''
    for i in range(10):
        password += random.choice(chars)
    return password

def get_curr_password():
    return curr_passwd


# def send_photo_file(chat_id, img):
#     TOKEN = '5275805360:AAGJb_iy7lyvxlHBNjwKKyLp85Ugm2L9fes'
#     URL = 'https://api.telegram.org/bot'
#     files = {'photo': open(img, 'rb')}
#     requests.post(f'{URL}{TOKEN}/sendPhoto?chat_id={chat_id}', files=files)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    global curr_passwd
    # curr_passwd = create_password()

    if message.text == "Привет":
        bot.send_message(message.from_user.id, "Привет, чем я могу тебе помочь?")
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Напиши привет")
    elif message.text == "Пароль":
        curr_passwd = create_password()
        bot.send_message(message.from_user.id, curr_passwd)
    # elif message.text == "Текущий пароль":
    #     bot.send_message(message.from_user.id, curr_passwd)
    else:
        bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")

bot.polling(none_stop=True, interval=0)