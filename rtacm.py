import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict


class RTACM:
    def __init__(self, model, autoencoder, difficulty_predictor, N, K, alpha=0.01, beta=0.001, num_tasks=9):
        self.model = model
        self.autoencoder = autoencoder
        self.difficulty_predictor = difficulty_predictor
        self.N = N
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.num_tasks = num_tasks
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=beta)
        self.loss_fn = nn.CrossEntropyLoss()

        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        self.difficulty_predictor_optimizer = optim.Adam(self.difficulty_predictor.parameters(), lr=0.001)

    def train_autoencoder(self, all_tasks):
        for epoch in range(100):
            total_loss = 0
            for task in all_tasks:
                data, _ = task['support']
                data = data.view(data.size(0), -1)
                decoded = self.autoencoder(data)
                loss = nn.MSELoss()(decoded, data)
                self.autoencoder_optimizer.zero_grad()
                loss.backward()
                self.autoencoder_optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Autoencoder Epoch {epoch + 1}, Loss: {total_loss / len(all_tasks):.4f}")

    def train_difficulty_predictor(self, auxiliary_tasks):
        for epoch in range(100):
            performances = torch.rand(len(auxiliary_tasks))

            total_loss = 0
            for task, performance in zip(auxiliary_tasks, performances):
                data, _ = task['support']
                data = data.view(1, data.size(0), -1)
                predicted_difficulty = self.difficulty_predictor(data)
                loss = nn.MSELoss()(predicted_difficulty.squeeze(), performance)
                self.difficulty_predictor_optimizer.zero_grad()
                loss.backward()
                self.difficulty_predictor_optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Difficulty Predictor Epoch {epoch + 1}, Loss: {total_loss / len(auxiliary_tasks):.4f}")

        with torch.no_grad():
            for i, (task, performance) in enumerate(zip(auxiliary_tasks[:5], performances[:5])):
                data, _ = task['support']
                data = data.view(1, data.size(0), -1)
                predicted = self.difficulty_predictor(data).item()
                print(f"Task {i}: Predicted = {predicted:.4f}, Actual = {performance.item():.4f}")

    def compute_task_similarity(self, task1, task2):
        data1, _ = task1['support']
        data2, _ = task2['support']
        data1 = data1.view(data1.size(0), -1)
        data2 = data2.view(data2.size(0), -1)
        encoded1 = self.autoencoder.encoder(data1)
        encoded2 = self.autoencoder.encoder(data2)
        similarity = torch.mean(torch.pow(encoded1.mean(0) - encoded2.mean(0), 2))
        return 1 / (similarity + 1e-5)

    def inner_loop(self, support_data, support_labels, task_similarity):
        fast_weights = OrderedDict(self.model.named_parameters())

        logits = self.model.forward(support_data)
        loss = self.loss_fn(logits, support_labels)
        grads = torch.autograd.grad(loss, self.model.parameters())

        fast_weights = OrderedDict(
            (name, param - self.alpha * grad * task_similarity)
            for ((name, param), grad) in zip(fast_weights.items(), grads)
        )

        return fast_weights

    def outer_loop(self, query_data, query_labels, fast_weights):
        logits = self.model.forward(query_data)
        loss = self.loss_fn(logits, query_labels)
        return loss

    def train(self, auxiliary_tasks, target_task, epochs):
        task_similarities = [self.compute_task_similarity(task, target_task) for task in auxiliary_tasks]

        difficulties = []
        for task in auxiliary_tasks:
            data, _ = task['support']
            data = data.view(1, data.size(0), -1)
            difficulty = self.difficulty_predictor(data).item()
            difficulties.append(difficulty)

        sorted_tasks = [x for _, x in sorted(zip(difficulties, auxiliary_tasks))]
        sorted_similarities = [x for _, x in sorted(zip(difficulties, task_similarities))]

        for epoch in range(epochs):
            meta_loss = 0
            for task, similarity in zip(sorted_tasks, sorted_similarities):
                support_data, support_labels = task['support']
                query_data, query_labels = task['query']

                support_indices = []
                for i in range(self.N):
                    class_indices = (support_labels == i).nonzero().squeeze()
                    if class_indices.dim() == 0:
                        support_indices.append(class_indices.item())
                    else:
                        selected_indices = class_indices[torch.randperm(len(class_indices))[:self.K]]
                        support_indices.extend(selected_indices.tolist())

                support_indices = torch.tensor(support_indices)
                support_data = support_data[support_indices]
                support_labels = support_labels[support_indices]

                fast_weights = self.inner_loop(support_data, support_labels, similarity)
                query_loss = self.outer_loop(query_data, query_labels, fast_weights)
                meta_loss += query_loss

            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Meta Loss: {meta_loss.item() / len(sorted_tasks):.4f}")

    def fine_tune(self, train_data, train_labels, epochs=10):
        optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

        for epoch in range(epochs):
            logits = self.model(train_data)
            loss = self.loss_fn(logits, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"Fine-tune Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    def evaluate(self, test_data, test_labels):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(test_data)
            loss = self.loss_fn(logits, test_labels)
            accuracy = (logits.argmax(dim=1) == test_labels).float().mean()
        return loss.item(), accuracy.item()