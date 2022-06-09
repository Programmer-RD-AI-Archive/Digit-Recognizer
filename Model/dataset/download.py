from Model import *


class Download:
    def __init__(self, test_size=0.125, shuffle=True, X_col="label", norm=255.0,g_path:str='./Model/dataset') -> None:
        self.data = pd.read_csv(f"{g_path}/raw/train.csv")
        self.test = pd.read_csv(f"{g_path}/raw/test.csv")
        self.sample_submission = pd.read_csv(f"{g_path}/raw/sample_submission.csv")
        self.test_size = test_size
        self.shuffle = shuffle
        self.X_col = X_col
        self.norm = norm

    def save_data_in_format(self) -> None:
        if "Img" not in os.listdir(f"{g_path}/data/"):
            os.makedirs("./Model/dataset/data/Img")
        n_data = {"Label": [], "Img Path": []}
        idx = 0
        for iter_idx, y_iter in enumerate(tqdm(self.y)):
            idx += 1
            n_data["Label"] = int(self.X[iter_idx])
            n_data["Img Path"] = f"{idx}.png"
            plt.imshow(y_iter.view(self.height, self.width, self.color_type), cmap="Greys")
            plt.savefig(f"./Model/dataset/data/Img/{idx}.png")
        n_data = pd.DataFrame(n_data)
        n_data.to_csv("./Model/dataset/data/data.csv", index=False)
        n_data.to_json(
            "./Model/dataset/data/data.json",
        )

    def save(self):
        torch.save(self.X, "./Model/dataset/data/X.pt")
        torch.save(self.y, "./Model/dataset/data/y.pt")
        torch.save(self.X_train, "./Model/dataset/data/X_train.pt")
        torch.save(self.X_test, "./Model/dataset/data/X_test.pt")
        torch.save(self.y_train, "./Model/dataset/data/y_train.pt")
        torch.save(self.y_test, "./Model/dataset/data/y_test.pt")

    def load(self) -> tuple:
        self.X = self.data[self.X_col]
        self.y = self.data.drop(self.X_col, axis=1)
        self.height = int(math.sqrt(len(self.y.columns)))
        self.width = int(math.sqrt(len(self.y.columns)))
        if self.height * self.width * 1 == len(self.y.columns):
            self.color_type = 1
        else:
            self.color_type = 3
        self.X = torch.from_numpy(np.array(self.X))
        self.y = torch.from_numpy(np.array(self.y))
        self.y = self.y.view(-1, self.color_type, self.height, self.width)
        self.y = self.y / self.norm
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, shuffle=self.shuffle
        )
        self.save()
        return (self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test)

    def test(self, model, img: torch.tensor, name: str) -> (np.array):
        preds = model(img)
        plt.figure(figsize=(12, 6))
        plt.title(preds)
        plt.imshow(img)
        plt.savefig(f"./Model/dataset/preds/{name}.png")
        plt.close()
        return np.array(cv2.imread(f"./Model/dataset/preds/{name}.png"))
