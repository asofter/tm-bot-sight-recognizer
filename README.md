# Tourist sights photo recognizer (Telegram bot)
This is Telegram bot written in Python which receives photo of some touristic sight and gives a description of this place with probability.

## Docker container
It supports running in docker container. Just run command ``docker-compose up -d``
or you can run simply by executing ``python bot.py`` in the **app** directory.

## Configuration

You need to rename file **config.sample.py** to **config.py** and put your bot key which you can get here: https://core.telegram.org/bots#botfather

## Retrain model
1. Uncomment second docker container in **docker-compose.yml** file.
2. Put new images in the **tf_files/sights/{LABEL_NAME}** directory.
3. In your project directory run: ``git clone https://github.com/tensorflow/tensorflow`` to clone tensorflow scripts.
4. Run new container using ``docker-compose up -d``
5. In this container (in the root directory) run 
```
python /tensorflow/tensorflow/examples/image_retraining/retrain.py \
 --bottleneck_dir=/tf_files/bottlenecks \
 --how_many_training_steps 500 \
 --output_graph=/tf_files/retrained_graph.pb \
 --output_labels=/tf_files/retrained_labels.txt \
 --image_dir /tf_files/sights
```
6. It will download inception model and retrain you model.
7. Copy files **retrained_graph.pb** and **retrained_labels.txt** from **tf_files** to **app/data**.
8. Comment tensorflow container till next training session.

## Reference
[Create a simple image classifier using Tensorflow](https://medium.com/@linjunghsuan/create-a-simple-image-classifier-using-tensorflow-a7061635984a)