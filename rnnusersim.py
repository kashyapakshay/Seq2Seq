from dataset_walker import dataset_walker
from prettyPrint import prettyPrint

import numpy as np
import matplotlib.pyplot as plt
import itertools

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation

class DialogueEncoder:
    def __init__(self):
        self.informable = ['area', 'food', 'name', 'pricerange']
        self.requestable = self.informable + ['addr', 'phone', 'postcode', 'signature']
        self.machineActs = ['affirm', 'bye', 'canthear', 'confirm-domain', 'negate', 'repeat', 'reqmore',
            'welcomemsg', 'canthelp', 'canthelp.missing_slot_value', 'canthelp.exception', 'expl-conf', 'impl-conf', 'inform',
            'offer', 'request', 'select', 'welcomemsg']

        self.userActs = ['ack', 'affirm', 'bye', 'hello', 'help', 'negate', 'null', 'repeat', 'reqalts',
            'reqmore', 'restart', 'silence', 'thankyou', 'confirm', 'deny', 'inform', 'request']

        self.inputShapeLen = 3 * len(self.informable) + len(self.requestable) + len(self.machineActs)
        self.outputShapeLen = len(self.userActs)

    def encodeUserActs(self, userActs):
        userActVector = [0] * len(self.userActs)
        for sem in userActs:
            userActVector[self.userActs.index(sem)] = 1

        return userActVector

    def encodeMachineActs(self, machineActs):
        machineActVector = [0] * len(self.userActs)
        for sem in machineActs:
            machineActVector[self.machineActs.index(sem)] = 1

        return machineActVector

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    dataset = dataset_walker("dstc2_dev", dataroot="data", labels=True)
    encoder = DialogueEncoder()

    X_train = []
    y_train = []

    blankMachineAct = encoder.encodeMachineActs([])

    for call in list(dataset):
        # print '\n----'
        # print '\nLOG: ', call.log["session-id"], '\n'

        dialogueHistory = [blankMachineAct]

        for turn, label in call:
            mact = encoder.encodeMachineActs([a['act'] for a in turn['output']['dialog-acts']])
            dialogueHistory.append(mact)
            uact = encoder.encodeUserActs([sem['act'] for sem in label['semantics']['json']])

            X_train.append(dialogueHistory[-2:-1] + [mact])
            y_train.append(uact)

            # print "\nSYSTEM:", turn['output']['transcript']
            # print "dialog acts:", [a['act'] for a in turn['output']['dialog-acts']]
            # print "Vector: ", mact
            #
            # print "\nUSER:", label['transcription']
            # print "dialog acts:", [sem['act'] for sem in label['semantics']['json']]
            # print "semantics:", label['semantics']['cam']
            # print "Vector: ", uact
            #
            # print '---\n'

    x_seventy = int(0.7 * len(X_train))
    y_seventy = int(0.7 * len(y_train))

    X_test = X_train[x_seventy:]
    y_test = y_train[y_seventy:]

    X_train = X_train[:x_seventy]
    y_train = y_train[:y_seventy]

    print np.array(X_train).shape

    model = Sequential()
    model.add(LSTM(output_dim=encoder.outputShapeLen, input_shape=(np.array(X_train).shape[1], np.array(X_train).shape[2]), activation='sigmoid'))
    # model.add(Dropout(0.05))
    # model.add(Dense(output_dim=encoder.outputShapeLen))
    # model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=16, nb_epoch=10)

    score = model.evaluate(X_test, y_test, batch_size=16)

    print score

    s = model.predict(X_test)

    correct = 0
    predictions = []

    # Decode predictions

    for j in range(0, len(s)):
        predicted = np.ndarray.tolist(1 / (1 + np.exp(-np.array(s[j]))))
        actual = y_test[j]

        for i in range(0, len(predicted)):
            if predicted[i] >= 0.579:
                predicted[i] = 1
            else:
                predicted[i] = 0

        predictions.append(predicted)

        if predicted == actual:
            correct += 1

        print '\n'
        print predicted
        print actual

    print "Acccuracy: ", (correct * 1.0 / len(s))

    # Confusion Matrix

    predictionsSet = set([tuple(a) for a in predictions])
    actualSet = set([tuple(a) for a in y_test])

    unionSet = predictionsSet | actualSet

    confusionMatrix = np.zeros((len(unionSet), len(unionSet))).tolist()

    for j in range(0, len(predictions)):
        predicted = predictions[j]
        actual = y_test[j]

        pi = list(unionSet).index(tuple(predicted))
        ai = list(unionSet).index(tuple(actual))

        confusionMatrix[pi][ai] += 1

    plt.figure()
    plot_confusion_matrix(np.array(confusionMatrix).astype(int), classes=[str(a) for a in range(0, len(unionSet))],
        title='Confusion matrix, without normalization')

    plt.show()
