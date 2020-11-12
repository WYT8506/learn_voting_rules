import matplotlib.pyplot as plt
import pickle
import numpy as np
if __name__ == "__main__":


    with open("num_data.txt", "rb") as fp:   # Unpickling
        num_data = pickle.load(fp)
    with open("consistency_satisfaction", "rb") as fp:   # Unpickling
        consistency_satisfactions = pickle.load(fp)
    with open("test_accuracy.txt", "rb") as fp:   # Unpickling
        test_accuracy = pickle.load(fp)


    no_active_no_aug_satisfactions = np.mean(np.array(consistency_satisfactions[3]),axis =0)
    no_active_no_aug_test_accuracy = np.mean(np.array(test_accuracy)[3],axis =0)
    active_no_aug_satisfactions = np.mean(np.array(consistency_satisfactions[1]),axis =0)
    active_no_aug_test_accuracy = np.mean(np.array(test_accuracy[1]),axis =0)
    active_aug_satisfactions = np.mean(np.array(consistency_satisfactions[0]),axis =0)
    active_aug_test_accuracy = np.mean(np.array(test_accuracy[0]),axis =0)
    no_active_aug_satisfactions = np.mean(np.array(consistency_satisfactions[2]),axis =0)
    no_active_aug_test_accuracy = np.mean(np.array(test_accuracy[2]),axis =0)
    active_aug_num_data = [100,200,300,400,500,600,700,800,900,1000]

    c1 = plt.plot(active_aug_num_data,active_aug_satisfactions,label ="active learning, data augmentation")
    c1 = plt.plot(active_aug_num_data,no_active_aug_satisfactions,label ="active learning, no data augmentation")
    c1 = plt.plot(active_aug_num_data,active_no_aug_satisfactions,label ="no active learning, data augmentation")
    c1 = plt.plot(active_aug_num_data,no_active_no_aug_satisfactions,label ="no active learning, no data augmentation")
    plt.legend()
    plt.show()

    c1 = plt.plot(active_aug_num_data,active_aug_test_accuracy,label ="active learning, data augmentation")
    c1 = plt.plot(active_aug_num_data,no_active_aug_test_accuracy,label ="active learning, no data augmentation")
    c1 = plt.plot(active_aug_num_data,active_no_aug_test_accuracy,label ="no active learning, data augmentation")
    c1 = plt.plot(active_aug_num_data,no_active_no_aug_test_accuracy,label ="no active learning, no data augmentation")
    print(len(active_aug_satisfactions))


    plt.legend()
    plt.show()