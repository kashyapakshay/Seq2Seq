from dataset_walker import dataset_walker
from prettyPrint import prettyPrint

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.preprocessing import sequence

import numpy as np

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

    for call in list(dataset)[:50]:
        print '\n----'
        print '\nLOG: ', call.log["session-id"], '\n'

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
            X_train.append(contextHistory + [contextVector])
            contextHistory.append(contextVector)
            # print X_train

            # print '\nVECTORS:'
            # print constraintVector
            # print requestVector

    print X_train[0]
    X_train = sequence.pad_sequences(X_train, maxlen=len(max(X_train)))[::-1]

    model = Sequential()
    model.add(LSTM(output_dim=outputShapeLen, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(Activation('sigmoid'))
    model.add(Dense(output_dim=outputShapeLen))
    model.add(Activation('sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=50, nb_epoch=20, validation_split=0.05)

    print '\nEvaluate:\n'

    s = model.predict(X_train, batch_size=32)

    for j in range(0, len(s)):
        predicted = np.ndarray.tolist(s[j])
        actual = y_train[j]

        for i in range(0, len(predicted)):
            if predicted[i] >= 0.5:
                predicted[i] = 1
            else:
                predicted[i] = 0

        print '\n'
        print predicted
        # print actual
