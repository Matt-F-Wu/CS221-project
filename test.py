import tensorflow as tf
import re
import data_utils
# New features
_DETECT_IDENTIFIER = re.compile(b"[\d\-_]")

# Since .. denotes parent directory, we treat is as a special file name
_DETECT_FILENAME = re.compile(b"([^.]+\.[^.]+|\.+)")

_FILENAME_SPLIT = re.compile(b"([;/\"'])")


def is_file(word):
  word = tf.compat.as_bytes(word)
  #contain_identifier = (not not _DETECT_IDENTIFIER.search(word))
  is_file_name = (not not _DETECT_FILENAME.match(word))
  # we can try learning more features as we proceed
  print(is_file_name)
  if is_file_name:
  	print(_FILENAME_SPLIT.split(word))

  print("=============")
  return is_file_name

'''
is_file('../../../')
is_file('somefile.txt;')
is_file('"somefile.txt";')
is_file('.')
is_file('..')
is_file('end.')
is_file('somefile..like')
'''
print(data_utils.custom_tokenizer(tf.compat.as_bytes("cd ../../../../")))
print(data_utils.custom_tokenizer(tf.compat.as_bytes("cd ..")))
print(data_utils.custom_tokenizer(tf.compat.as_bytes("find / -name somefile.txt")))
print(data_utils.custom_tokenizer(tf.compat.as_bytes("go up 3 levels")))
print(data_utils.custom_tokenizer(tf.compat.as_bytes("give me happy3 please")))
print(data_utils.is_file_name(tf.compat.as_bytes("happy3")))