from dataset_walker import dataset_walker
from prettyPrint import prettyPrint

import tensorflow as tf

if __name__ == '__main__':
    dataset = dataset_walker("dstc2_dev", dataroot="data", labels=True)

    informable = ['area', 'food', 'name', 'pricerange']
    requestable = informable + ['addr', 'phone', 'postcode', 'signature']

    for call in dataset:
        if call.log["session-id"] == "voip-f246dfe0f2-20130328_161556":
            constraintVector = [0] * len(informable)
            requestVector = [0] * len(requestable)

            constraintValues = [''] * len(informable)

            for turn, label in call:
                inconsistencyVector = [0] * (2 * len(informable))

                print '\n----'

                print "\nSYSTEM:", turn['output']['transcript']
                print "dialog acts:", [a['act'] for a in turn['output']['dialog-acts']]

                machineMentioned = []
                for act in turn['output']['dialog-acts']:
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

                print 'INCONSISTENCY: ', inconsistencyVector

                # --- DO TRAINING HERE ---
                # This turn's machine response corresponds to last turns user action.
                # So here, we have the correctly built inconsistency and constraint vectors.

                # ...

                print "\nUSER:", label['transcription']
                print "dialog acts:", [sem['act'] for sem in label['semantics']['json']]
                print "semantics:", label['semantics']['cam']

                print label['semantics']['json']

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

                print '\nVECTORS:'
                print constraintVector
                print requestVector
