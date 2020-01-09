#encoding:utf-8
from __future__ import print_function
from text_model import *
from loader import *
from sklearn import metrics
import sys
import os
import time
from datetime import timedelta

def evaluate(sess, x_, y_):#计算loss和acc
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len

def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob:keep_prob
    }
    return feed_dict

def train():
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = './tensorboard/textcnn' 
    save_dir = './checkpoints/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')

    print("Loading training and validation data...")
    start_time = time.time()
    x_train, y_train = process_file(config.train_filename, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file2(config.val_filename, word_to_id, cat_to_id, config.seq_length)
    print("Time cost: %.3f seconds...\n" % (time.time() - start_time))

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    best_val_accuracy = 0
    last_improved = 0  # record global_step at best_val_accuracy
    require_improvement = 1000  # break training if not having improvement over 1000 iter
    flag=False

    for epoch in range(config.num_epochs):
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        start = time.time()
        print('Epoch:', epoch + 1)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.keep_prob)
            _, global_step, train_summaries, train_loss, train_accuracy = session.run([model.optim, model.global_step,
                                                                                    merged_summary, model.loss,
                                                                                    model.acc], feed_dict=feed_dict)
            if global_step % config.print_per_batch == 0:
                end = time.time()
                val_loss, val_accuracy = evaluate(session, x_val, y_val)
                writer.add_summary(train_summaries, global_step)

                # If improved, save the model
                if val_accuracy > best_val_accuracy:
                    saver.save(session, save_path)
                    best_val_accuracy = val_accuracy
                    last_improved=global_step
                    improved_str = '*'
                else:
                    improved_str = ''
                print("step: {},train loss: {:.3f}, train accuracy: {:.3f}, val loss: {:.3f}, val accuracy: {:.3f},training speed: {:.3f}sec/batch {}\n".format(
                        global_step, train_loss, train_accuracy, val_loss, val_accuracy,
                        (end - start) / config.print_per_batch,improved_str))
                start = time.time()

            if global_step - last_improved > require_improvement:
                print("No optimization over 1000 steps, stop training")
                flag = True
                break
        if flag:
            break
        config.lr *= config.lr_decay

if __name__ == '__main__':
    print('Configuring CNN model...')
    config = TextConfig()#该函数在text_model文件中，用于定义cnn参数以及数据位置
    filenames = [config.train_filename, config.test_filename, config.val_filename]#获取训练数据、测试数据以及验证集
    if not os.path.exists(config.vocab_filename):#如果不存在词汇列表文件，则调用build_vocab（在loader文件中）创建
        build_vocab(filenames, config.vocab_filename, config.vocab_size)

    #read vocab and categories
    categories,cat_to_id = read_category()#在loader文件中，得到类别与类别索引字典
    words,word_to_id = read_vocab(config.vocab_filename)#在loader文件中，得到单词与单词索引字典
    config.vocab_size = len(words)

    # trans vector file to numpy file
    if not os.path.exists(config.vector_word_npz):
        export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)
    config.pre_trianing = get_training_word2vec_vectors(config.vector_word_npz)

    model = TextCNN(config)
    train()
