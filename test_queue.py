import sugartensor as tf

q = tf.FIFOQueue(3, 'int32')
init = q.enqueue_many(([0, 10, 20], ))
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print v