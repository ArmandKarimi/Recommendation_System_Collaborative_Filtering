import torch
import torch.nn.functional as F

def model_test(model, test_loader, loss_fn, device):
    
    test_loss = 0
    total_samples = 0
    all_predictions = []
    all_ratings = []
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for batch in test_loader:
            users, movies, ratings = batch
            users = users.to(device)
            movies = movies.to(device)
            ratings = ratings.to(device)

            predictions = model(users, movies)
            loss = loss_fn(predictions, ratings)
            test_loss += loss.item()

            # Collect predictions and true ratings for RMSE
            all_predictions.extend(predictions.cpu().numpy())
            all_ratings.extend(ratings.cpu().numpy())
            total_samples += len(ratings)

    # Compute RMSE
    mse = F.mse_loss(torch.tensor(all_predictions), torch.tensor(all_ratings)).item()
    rmse = mse ** 0.5

    print(f"Test MSE: {mse:.4f}, Test RMSE: {rmse:.4f}")

    return rmse  # Return RMSE for further analysis if needed
