#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Bot which reply with place name of Chernivtsi

from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, RegexHandler,
                          ConversationHandler)

import logging
import hashlib
import tensorflow as tf, sys
import time
import pandas as pd
import os
import config

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

PHOTO = range(1)


def start(bot, update):
    update.message.reply_text(
        'Привіт. Я бот, який вміє вгадувати пам\'ятки в місті Чернівці.\n\n'
        'Командою /cancel можна зупинити нашу переписку.\n\n'
        'Для того, щоб почати, сфотографуй пам\'ятку і відправ мені.',
        reply_markup=ReplyKeyboardRemove())

    return PHOTO


def photo(bot, update):
    user = update.message.from_user
    photo_file = bot.get_file(update.message.photo[-1].file_id)
    current_time = str(time.time())
    image_path = 'images/' + current_time + '.jpg'
    photo_file.download(image_path)
    logger.info("Photo of %s: %s", user.first_name, image_path)
    update.message.reply_text('Супер. Тепер почекай, щоб я написав тобі, що я знаю про це місце.')

    recognizer = recognize_image(image_path)
    if len(recognizer) > 0:
        name = recognizer[0]['name']
        percentage = str("{0:.2f}".format(recognizer[0]['score'] * 100))
        info = sight_info(name)
        if 'title' in info:
            name = info['title']

        update.message.reply_text(
            'Я думаю, що це ' + name + ' з вірогідністю '+percentage+'%.'
        )
    else:
        update.message.reply_text('На жаль, я нічого не знайшов про це місце.')

    os.remove(image_path)

    return ConversationHandler.END


label_lines = []
graph_def = None


def init_recognizer():
    global label_lines, graph_def

    # Loads label file, strips off carriage return
    if len(label_lines) == 0:
        label_lines = [line.rstrip() for line
                       in tf.gfile.GFile("data/retrained_labels.txt")]

    # Unpersists graph from file
    if graph_def is None:
        with tf.gfile.FastGFile("data/retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')


def recognize_image(image_path):
    global label_lines

    init_recognizer()

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Feed the image_data as input to the graph and get first prediction
    result = []
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]

            if score > config.THRESHOLD:
                diction = {"score": score, "name": human_string}
                result.append(diction)

    return result


def sight_info(sight_id):
    sights = pd.read_csv("sights.csv", sep=';', encoding='utf-8')
    sight_info = sights[(sights['id'] == sight_id)].values

    result = {}
    if len(sight_info) > 0:
        result['title'] = sight_info[0][1]

    return result


def cancel(bot, update):
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text('Бувай! Надіюсь ми ще поспілкуємось.',
                              reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    # Create the EventHandler and pass it your bot's token.
    updater = Updater(config.TELEGRAM_API_KEY)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Add conversation handler with the states PHOTO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            PHOTO: [MessageHandler(Filters.photo, photo)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )

    dp.add_handler(conv_handler)

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()