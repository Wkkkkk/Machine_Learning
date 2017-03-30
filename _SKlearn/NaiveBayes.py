from Tools.Bases import ClassifierBase
from Tools.Metas import SKCompatibleMeta

import sklearn.naive_bayes as nb


class SKMultinomialNB(nb.MultinomialNB, ClassifierBase, metaclass=SKCompatibleMeta):
    pass


class SKGaussianNB(nb.GaussianNB, ClassifierBase, metaclass=SKCompatibleMeta):
    pass
