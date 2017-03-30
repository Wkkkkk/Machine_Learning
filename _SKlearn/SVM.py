from Tools.Bases import ClassifierBase
from Tools.Metas import SKCompatibleMeta

from sklearn.svm import SVC


class SKSVM(SVC, ClassifierBase, metaclass=SKCompatibleMeta):
    pass
