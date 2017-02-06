from dataset_walker import dataset_walker
from prettyPrint import prettyPrint

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, GRU, Embedding, TimeDistributed
from keras.preprocessing import sequence

import numpy as np
import matplotlib.pyplot as plt
import itertools

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

    informable = ['area', 'food', 'name', 'pricerange']
    requestable = informable + ['addr', 'phone', 'postcode', 'signature']
    machineActs = ['affirm', 'bye', 'canthear', 'confirm-domain', 'negate', 'repeat', 'reqmore',
        'welcomemsg', 'canthelp', 'canthelp.missing_slot_value', 'canthelp.exception', 'expl-conf', 'impl-conf', 'inform',
        'offer', 'request', 'select', 'welcomemsg']

    userActs = ['ack', 'affirm', 'bye', 'hello', 'help', 'negate', 'null', 'repeat', 'reqalts',
        'reqmore', 'restart', 'silence', 'thankyou', 'confirm', 'deny', 'inform', 'request']

    inputShapeLen = 3 * len(informable) + len(requestable) + len(machineActs)
    outputShapeLen = len(userActs)

    # contextHistory = []

    X_train = []
    y_train = []

    blankMachineAct = [0] * inputShapeLen
    for call in list(dataset):
        dialogueHistory = [blankMachineAct, blankMachineAct]

        # print '\n----'
        # print '\nLOG: ', call.log["session-id"], '\n'

        constraintVector = [0] * len(informable)
        requestVector = [0] * len(requestable)
        userActVector = [0] * len(userActs)

        constraintValues = [''] * len(informable)

        contextHistory = []

        for turn, label in call:
            machineActVector = [0] * len(machineActs)
            inconsistencyVector = [0] * (2 * len(informable))

            # print "\nSYSTEM:", turn['output']['transcript']
            # print "dialog acts:", [a['act'] for a in turn['output']['dialog-acts']]

            machineMentioned = []
            for act in turn['output']['dialog-acts']:
                # Build machine act vector
                machineActVector[machineActs.index(act['act'])] = 1

                if act['act'] == 'inform':
                    for slot in act['slots']:
                        # If the machine misunderstood the user specified value for constraint, set it to 0
                        if slot[0] in informable:
                            machineMentioned.append(slot[1])

                            if constraintValues[informable.index(slot[0])] != '' and slot[1] != constraintValues[informable.index(slot[0])]:
                                inconsistencyVector[4 + informable.index(slot[0])] = 1
                                constraintVector[informable.index(slot[0])] = 0

            for i in range(0, len(constraintValues)):
                if constraintValues[i] != '' and constraintValues[i] != 'dontcare':
                    if constraintValues[i] not in machineMentioned:
                        inconsistencyVector[i] = 1
                        constraintVector[i] = 0

            for i in range(0, len(constraintValues)):
                if inconsistencyVector[i] != 1 and inconsistencyVector[i+4] != 1 and constraintValues[i] != '':
                    constraintVector[i] = 1

            # print 'INCONSISTENCY: ', inconsistencyVector
            # print 'MACHINE: ', machineActVector

            # Build Context Vector - concatenate all vectors
            contextVector = machineActVector + inconsistencyVector + constraintVector + requestVector

            y_train.append(userActVector)
            # print "\nUSER:", label['transcription']
            # print "dialog acts:", [sem['act'] for sem in label['semantics']['json']]
            # print "semantics:", label['semantics']['cam']
            #
            # print label['semantics']['json']

            # Build Request Vector
            for request in label['requested-slots']:
                requestVector[requestable.index(request)] = 1

            # Build Constraint Vector
            for semantic in label['semantics']['json']:
                if semantic['act'] == 'inform':
                    for slot in semantic['slots']:
                        if slot[0] in informable and slot[1] != 'dontcare':
                            # constraintVector[informable.index(slot[0])] = 1
                            # Last specified constraint value for each contraint slot
                            constraintValues[informable.index(slot[0])] = slot[1]

            # Encode User acts as one-hot encoding
            userActVector = [0] * len(userActs)
            for sem in label['semantics']['json']:
                userActVector[userActs.index(sem['act'])] = 1

            contextVectorStr = ''.join([str(x) for x in contextVector])
            userActVectorStr = ''.join([str(x) for x in userActVector])

            # print 'CONTEXT VECTOR: ', contextVectorStr
            # print 'USER ACT VECTOR: ', userActVectorStr

            # print contextHistory
            bufferHistory = contextHistory + [contextVector]
            # print bufferHistory[-4:-1]
            dialogueHistory.append(contextVector)
            X_train.append(dialogueHistory[-3:-1] + [contextVector])
            contextHistory.append(contextVector)
            # print X_train

            # print '\nVECTORS:'
            # print constraintVector
            # print requestVector

    # X_train = sequence.pad_sequences(X_train, maxlen=3)
    X_train = np.array(X_train)
    x_seventy = int(0.7 * len(X_train))
    y_seventy = int(0.7 * len(y_train))

    X_test = X_train[x_seventy:]
    y_test = y_train[y_seventy:]

    X_train = X_train[:x_seventy]
    y_train = y_train[:y_seventy]

    X_shape = np.array(X_train).shape
    y_shape = np.array(y_train).shape

    model = Sequential()
    model.add(LSTM(output_dim=outputShapeLen, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
    model.add(GRU(output_dim=outputShapeLen, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=False))
    model.add(Dense(output_dim=outputShapeLen, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=16, nb_epoch=10)

    print '\nEvaluate:\n'

    s = model.predict(X_test)
    print model.evaluate(X_test, y_test, batch_size=100, verbose=1)

    # EVALUATE

    correct = 0
    predictions = []

    precision = 0.0
    recall = 0.0

    # Decode predictions

    for j in range(0, len(s)):
        predicted = (1 / (1 + np.exp(-np.array(s[j]))))
        # predicted -= np.amin(predicted)
        # predicted = np.round(predicted, 1)
        predicted = np.ndarray.tolist(predicted)

        # predicted = s[j]
        actual = y_test[j]

        localPrecision = 0.0
        localRecall = 0.0

        for i in range(0, len(predicted)):
            if predicted[i] >= 0.58:
                predicted[i] = 1
            else:
                predicted[i] = 0

        if predicted == actual:
            correct += 1

        for pos in range(len(predicted)):
            if predicted[pos] == 1:
                if actual[pos] == 1:
                    localPrecision += 1

        for pos in range(len(actual)):
            if actual[pos] == 1:
                if predicted[pos] == 1:
                    localRecall += 1

        precisionCount = predicted.count(1)
        localPrecisionAvg = 0.0
        if precisionCount != 0:
            localPrecisionAvg += localPrecision / precisionCount

        recallCount = actual.count(1)
        localRecallAvg = 0.0
        if recallCount != 0:
            localRecallAvg = localRecall / recallCount

        precision += localPrecisionAvg
        recall += localRecallAvg

        predictions.append(predicted)

        print '\n'
        print '[%d]' % (j)
        print 'predicted: ', predicted
        print 'actual: ', actual
        print 'Local Precision: %f' % (localPrecisionAvg)
        print 'Local Recall: %f' % (localRecallAvg)

    print 'Accuracy: %f' % (correct * 1.0 / len(s))
    print 'Precision: %f' % (precision / len(s))
    print 'Recall: %f' % (recall / len(s))

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
