import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import seaborn as sns
import matplotlib.pyplot as plt


class pcaSwapBuilder(object):
    def __init__(self, dataDF) -> None:
        super().__init__()
        self.dataSet = dataDF
        self.xLoadings = None
        self.xScores = None
        self.predicted = None
        self.residual = None
        self.nResidual = None

    def get_dataSet(self):
        return self.dataSet

    def pcaModelBuild(self):
        modelDataSet = self.get_dataSet()
        scal = StandardScaler()
        scal.fit(modelDataSet)
        normalizedModelDataSet = scal.transform(modelDataSet)
        pcaModel = sklearnPCA(n_components=3)
        self.xScores = pcaModel.fit_transform(normalizedModelDataSet)
        self.xLoadings = pcaModel.components_
        normalizedPredicted = pcaModel.inverse_transform(self.xScores)
        predicted = scal.inverse_transform(normalizedPredicted)
        self.predicted = pd.DataFrame(
            predicted, index=modelDataSet.index, columns=modelDataSet.columns
        )
        residual = modelDataSet - self.predicted
        self.residual = residual
        nResidual = (residual - residual.mean()) / residual.std()
        self.nResidual = nResidual

    def get_Normalized_Residual_Data(self):
        return self.nResidual

    def get_Predicted_Prices(self):
        return self.predicted


if __name__ == "__main__":
    pass
