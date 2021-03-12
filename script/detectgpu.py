import tensorflow as tf

print("Checking for GPU in list of system devices:")
print(tf.config.list_physical_devices("GPU"))
