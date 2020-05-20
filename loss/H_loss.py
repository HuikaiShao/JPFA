import tensorflow as tf


def hash_loss(feature,label_batch,batch_size,omega_size):

    archer_code,sabor_code = tf.split(feature,[omega_size,batch_size-omega_size],axis = 0)
    archer_label,sabor_label = tf.split(label_batch,[omega_size,batch_size-omega_size],axis = 0)
    
    archer_matrix = tf.matmul(archer_code,tf.transpose(archer_code))
    sabor_matrix = tf.matmul(sabor_code,tf.transpose(sabor_code))

    archer_Similarity = tf.matmul(archer_label,tf.transpose(archer_label))
    sabor_Similarity = tf.matmul(archer_label,tf.transpose(sabor_label))
    archer_diag = tf.transpose(tf.reshape(tf.tile(tf.diag_part(archer_matrix),[omega_size]),[omega_size,omega_size]))
    archer_sabor_diag = tf.transpose(tf.reshape(tf.tile(tf.diag_part(archer_matrix),[batch_size-omega_size]),[batch_size-omega_size,omega_size]))
    sabor_diag = tf.reshape(tf.tile(tf.diag_part(sabor_matrix),[omega_size]),[omega_size,batch_size-omega_size])

    archer_distance = archer_diag + tf.transpose(archer_diag) - 2*archer_matrix
    sabor_distance = sabor_diag + archer_sabor_diag - 2*tf.matmul(archer_code,tf.transpose(sabor_code))
    archer_loss = tf.reduce_mean(1/2*archer_Similarity*archer_distance + 1/2*(1-archer_Similarity)*tf.maximum(180-archer_distance,0))
    sabor_loss = tf.reduce_mean(1/2*sabor_Similarity*sabor_distance + 1/2*(1-sabor_Similarity)*tf.maximum(180-sabor_distance,0))
    loss = archer_loss + sabor_loss

    return loss
