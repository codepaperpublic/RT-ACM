from models import Autoencoder, DifficultyPredictor, BearingNet
from rtacm import RTACM  # 更新了导入
from data_generator import generate_task_data


def main():
    input_size = 100
    hidden_size = 64
    output_size = 8
    num_tasks = 10
    N = 8
    K = 1
    query_samples = 15
    epochs = 1000

    model = BearingNet(input_size, hidden_size, output_size)
    autoencoder = Autoencoder(input_size, 32)
    difficulty_predictor = DifficultyPredictor(input_size, 64)

    rtacm = RTACM(model, autoencoder, difficulty_predictor, N=N, K=K, alpha=0.01, beta=0.001,
                  num_tasks=num_tasks - 1)  # 更新了类名

    all_tasks = []
    for _ in range(num_tasks):
        support_set, query_set = generate_task_data(N, K, query_samples, input_size)
        all_tasks.append({
            'support': support_set,
            'query': query_set
        })

    target_task = all_tasks.pop()
    auxiliary_tasks = all_tasks

    rtacm.train_autoencoder(all_tasks)
    rtacm.train_difficulty_predictor(auxiliary_tasks)
    rtacm.train(auxiliary_tasks, target_task, epochs)

    train_data, train_labels = target_task['support']
    rtacm.fine_tune(train_data, train_labels, epochs=50)

    test_data, test_labels = target_task['query']
    test_loss, test_accuracy = rtacm.evaluate(test_data, test_labels)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()