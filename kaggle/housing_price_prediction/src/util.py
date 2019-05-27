'''
Some helpful functions and classes for evaluating
model selection.
'''

class ParamScore:
    '''
    Might as well be a C++ struct.
    '''
    def __init__(self, classifier, score, stdev=0.0):
        self.classifier = classifier
        self.score = score
        self.stdev = stdev
    
    def __repr__(self):
        return repr((self.classifier, self.score, self.stdev))
     
