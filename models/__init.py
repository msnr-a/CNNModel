import os
import re
import numpy as np
from datetime import datetime as time
from tensorflow.keras.models import Sequential

class CNNModel(Sequential):
    
    __total_epochs = 0

    @property
    def total_epochs(self):
        """int : インスタンスが生成されてから実行されたトレーニング数を取得する。\n"""
        return self.__total_epochs

    def __recompile(self):
        
        # コンパイル確認
        if not self._is_compiled:
            raise RuntimeError("You must compile a model before training.")
 
        # コンパイルの設定
        if self.compiled_metrics == None:
            self.compile(
                loss=self.loss,
                optimizer=self.optimizer)
        else:
            self.compile(
                loss=self.loss,
                optimizer=self.optimizer,
                metrics=self.compiled_metrics._user_metrics)

    def defit(self, x_train, y_train, epochs=1, batch_size=128):
        """学習をエポック数回だけ実行する。
 
            Parameters
            ----------
            x_train : numpy.ndarray\n
                学習データ\n
            y_train : numpy.ndarray\n
                正解データ\n
            epochs :  int\n
                エポック数。学習を繰り返す回数を指定する。\n
                default : 1\n
                validation : >0\n
        """
        # コンパイルされていない場合は実行
        self.__recompile()

        for i in range(epochs):

            # 損失関数
            loss = []

            # 総EPOCK数を++
            self.__total_epochs += 1

            # 一度に学習するデータ数
            nbat = len(x_train) // batch_size

            # 学習データ取得
            x_batch = x_train[(i * nbat):(i * nbat + batch_size)]
            y_batch = y_train[(i * nbat):(i * nbat + batch_size)]

            for j in range(batch_size):

                # 進捗表示
                q, mod = divmod(j, (batch_size // 30))
                if mod == 0: print(f"\r{self.__total_epochs} [{'='*q}{'>' if q <= 30 else ''}{'.'*(30 - q)}]",end="")

                # 学習実行
                loss.append(self.train_on_batch(x_batch, y_batch))

            loss = np.array(loss)
            print(f" loss:{np.average(loss[:,0]):.4f}, acc:{np.average(loss[:, 1]) * 100:.4f}")


    def saveWeights(self, path):
        """重みデータを保存する。
 
            Parameters
            ----------
            path : string\n
                保存するファイルのパス、またはディレクトリを指定する。\n
                ディレクトリを指定した場合、保存ファイル名は{yyyymmddhhmmss}_wgt_{total_epochs}.h5となる。\n
            returns
            -------
            filepath : string\n
                保存したファイルのパスを返す。
        """

        # ディレクトリの場合はテンプレで保存
        if os.path.isdir(path):
            filename = os.path.join(path, f"{time.today():%Y%m%d%H%M%S}_wgt_{self.total_epochs}.h5")
        else:
            filename = path
        
        # 再コンパイルして保存
        self.__recompile()
        self.save_weights(filename)
 
        # 保存したファイルパスを戻す
        return filename
 
    def loadWeights(self, path):
        """重み￥データを読込む。
 
            Parameters
            ----------
            path : string\n
                読込むファイルのパス、またはディレクトリを指定する。\n
                ディレクトリを指定した場合、ファイル名が\d{14}_wgt_\d+\.h5であるファイルの内、ソート後の順が最後尾のファイルを使用する。\n
            returns
            -------
            filepath : string\n
                保存したファイルのパスを返す。
            raises
            ------
            FileNotFoundError
                読込むファイルが存在しない場合発生する。
        """
 
        # 読込みファイル名の正規表現
        LOAD_FILE_NAME = "\d{14}_wgt_(\d+)\.h5"
 
        # ディレクトリの場合はLOAD_FILE_NAMEをソート後の最後尾
        if os.path.isdir(path):
 
            # ファイルのみ取得
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            if len(files) == 0:
                # raise FileNotFoundError(f"No such file or directory: '{path}/*'")
                print("最初から学習開始")
                return False
            
            # 学習ファイルを検索
            files = [f for f in files if re.match(LOAD_FILE_NAME, f)]
            if len(files) == 0:
                # raise FileNotFoundError(f"No such file or directory: '{path}/*'")
                print("最初から学習開始")
                return False
            
            # 最新のファイル（最後尾）を取得
            files = sorted(files)
            filename = os.path.join(path, files[-1])
 
            # epoch番号取得
            num = re.match(LOAD_FILE_NAME, files[-1])
            self.__total_epochs = int(num.group(1))
            
        else:
            filename = path
 
        # 再コンパイルして読込み
        self.__recompile()
        self.load_weights(filename)
 
        # 読込んだファイルパスを戻す
        print("前回のチェックポイントから再開")
        return filename
