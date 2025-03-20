import os 
import torch
import torch.nn as nn

def model_train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):

    model.to(device)  # Move model to device

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        train_loss = 0
        
        for batch in train_loader:
            users, movies, ratings = batch
            users = users.to(device)
            movies = movies.to(device)
            ratings = ratings.to(device)    

            optimizer.zero_grad()

            # Forward pass: predict ratings
            predictions = model(users, movies)
            
            # Compute the loss
            loss = loss_fn(predictions, ratings)
            loss.backward()  # Backpropagate the error
            optimizer.step()  # Update the weights
            
            train_loss += loss.item()
        
        # Validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                users, movies, ratings = batch
                users = users.to(device)
                movies = movies.to(device)
                ratings = ratings.to(device)

                predictions = model(users, movies) 
                loss = loss_fn(predictions, ratings)
                val_loss += loss.item()

        # Print epoch loss
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

    return model