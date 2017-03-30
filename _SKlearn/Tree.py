from Tools.Bases import ClassifierBase
from Tools.Metas import SKCompatibleMeta

from sklearn.tree import DecisionTreeClassifier


class SKTree(DecisionTreeClassifier, ClassifierBase, metaclass=SKCompatibleMeta):
    pass
