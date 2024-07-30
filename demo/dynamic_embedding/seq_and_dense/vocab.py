import tensorflow as tf
import tensorflow_datasets as tfds


def get_dataset():
  ds = tfds.load("movielens/1m-ratings",
                 split="train",
                 data_dir="~/dataset",
                 download=True)
  ds = ds.map(
      lambda x: {
          "movie_genres":
              tf.cast(x["movie_genres"][0], tf.int64),
          "user_occupation_label":
              tf.cast(x["user_occupation_label"], tf.int64),
      })

  max_min_dict = {
      "max_movie_genres": tf.int64.min,
      "min_movie_genres": tf.int64.max,
      "max_user_occupation_label": tf.int64.min,
      "min_user_occupation_label": tf.int64.max
  }

  for sample in ds:
    max_min_dict["max_movie_genres"] = tf.maximum(
        max_min_dict["max_movie_genres"], sample["movie_genres"])
    max_min_dict["min_movie_genres"] = tf.minimum(
        max_min_dict["min_movie_genres"], sample["movie_genres"])
    max_min_dict["max_user_occupation_label"] = tf.maximum(
        max_min_dict["max_user_occupation_label"],
        sample["user_occupation_label"])
    max_min_dict["min_user_occupation_label"] = tf.minimum(
        max_min_dict["min_user_occupation_label"],
        sample["user_occupation_label"])

  return max_min_dict


max_min_stats = get_dataset()
# max_movie_genres 19 min_movie_genres 0 max_user_occupation_label 21 min_user_occupation_label 0
print(max_min_stats)
