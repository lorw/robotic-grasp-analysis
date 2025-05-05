#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 11:53:19 2025

@author: ro-bo-ds
"""

import pandas as pd

# ------[0.] DOWNLOAD & LOAD DATASET ------

# load data into a df
df = pd.read_csv("my_rl_learning_data1.csv")
print("Data shape:", df.shape)
df.head()


# ------[1.] CLEAN AND INSPECT ------

# count missing values by column
df.isna().sum()
# convert to numeric, errors to NaN
numeric_cols = [
    "value_func_loss", "surrogate_loss", "mean_total_reward",
    "episode_reward_reaching_object", "episode_reward_lifting_object",
    "episode_reward_obj_goal_tracking", "episode_reward_obj_goal_tracking_fine",
    "episode_reward_action_rate", "episode_reward_joint_vel",
    "metrics_pos_error", "metrics_ori_error", "mean_episode_length",
    "computation_steps_s", "total_timesteps", "iteration_time_s", "total_time_s"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# drop missing columns
df = df.dropna(axis=1, how='all')
df.describe()


# ----------[3.] EDA ------------- #

import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(df["iteration"], df["episode_reward_reaching_object"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Reaching Object Reward")
plt.title("Reaching Reward vs. Iteration")
plt.show()

df["combined_reward"] = (
    df["episode_reward_reaching_object"] +
    df["episode_reward_lifting_object"] +
    df["episode_reward_obj_goal_tracking"] +
    df["episode_reward_obj_goal_tracking_fine"] +
    df["episode_reward_action_rate"] +
    df["episode_reward_joint_vel"]
)

plt.plot(df["iteration"], df["combined_reward"], marker='o')
plt.xlabel("Iteration")
plt.ylabel("Combined Reward")
plt.title("Combined Reward vs. Iteration")
plt.show()

# overall, these plots point to two distinct phases in the run:
# phase 1: iterations 0–400, with small pos rewards or near zero for reaching and combined.
# phase 2: iters > 400, with significantly neg combined reward that gradually improve toward the end.



# ---DO Correlation Matrix & DISTRIBUTION ANALYSIS---:
    
import seaborn as sns
import matplotlib.pyplot as plt

cols_for_corr = [
    "episode_reward_reaching_object",
    "episode_reward_lifting_object",
    "episode_reward_obj_goal_tracking",
    "episode_reward_obj_goal_tracking_fine",
    "episode_reward_action_rate",
    "episode_reward_joint_vel",
    "metrics_pos_error",
    "metrics_ori_error",
    "combined_reward"
]

df_corr = df[cols_for_corr].dropna()

corr_matrix = df_corr.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Key Reward & Error Metrics")
plt.show()

# this identifies which metrics drive combined reward most

df[["episode_reward_reaching_object", "combined_reward"]].hist(
    bins=20, figsize=(10,4), layout=(1,2)  # 2, side-by-side
)
plt.tight_layout()
plt.show()

# combined_reward ranges from about -8 to 0


# --- EXPLORE "PHASES" in our data by looking at slices closer, esp around 400 iterations (big dip happens here, then climbs): ---

# slice the df's
phase_cut = 400
df_phase1 = df[df["iteration"] <= phase_cut]
df_phase2 = df[df["iteration"] > phase_cut]
print("Phase 1 shape:", df_phase1.shape)
print("Phase 2 shape:", df_phase2.shape)

# compare:
print("Phase 1 combined_reward mean:", df_phase1["combined_reward"].mean())
print("Phase 2 combined_reward mean:", df_phase2["combined_reward"].mean())

# plot results side-by-side:
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(df_phase1["iteration"], df_phase1["combined_reward"], marker='o')
plt.title("Phase 1 (iter <= 400)")
plt.xlabel("Iteration")
plt.ylabel("Combined Reward")

plt.subplot(1,2,2)
plt.plot(df_phase2["iteration"], df_phase2["combined_reward"], marker='o')
plt.title("Phase 2 (iter > 400)")
plt.xlabel("Iteration")

plt.tight_layout()
plt.show()

# take ROLLING AVGS (for smoothing, since we see it going up and down a lot):

window_size = 50
df["combined_reward_rolling"] = (
    df["combined_reward"]
    .rolling(window_size, min_periods=1)  # so the first few points still show something
    .mean()
)

plt.figure(figsize=(8,4))
plt.plot(df["iteration"], df["combined_reward"], alpha=0.4, label="Combined (raw)")
plt.plot(df["iteration"], df["combined_reward_rolling"], color="red", label=f"Rolling Mean ({window_size})")
plt.xlabel("Iteration")
plt.ylabel("Combined Reward")
plt.title("Combined Reward (Rolling Avg)")
plt.legend()
plt.show()

# this gave us a better, smoother, image/visualization of the averages. we can see the avgs' trends better in the line graph.



# ----------[4.] REGRESSION ------------- #

# how to find out how different cols predict the combine dreward? do a regression, using RANDOM FOREST REGRESSOR:

from sklearn.model_selection import train_test_split

target = "combined_reward"
predictors = [
    "episode_reward_reaching_object",
    "episode_reward_lifting_object",
    "episode_reward_action_rate",
    "metrics_pos_error",
    "metrics_ori_error"
]

df_reg = df.dropna(subset=[target] + predictors)
X = df_reg[predictors]
y = df_reg[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)




# --- DO RANDOM FOREST REGRESSOR: ---

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R^2:", r2)
print("MSE:", mse)

# feature importance seen here:
for feat, imp in zip(predictors, rf.feature_importances_):
    print(f"{feat}: {imp:.3f}")
    
# INTERPRET/EXPLAIN RESULTS?:
# R² near 1.0: The model fits well.
# R² near 0.0: The model is not capturing much variance.
# negative R²: The model is worse than a simple, mean-based prediction.
# remember, feature importances show which cols had the biggest effect on predicting combined reward.


# check residuals--why?--to see if our model systematically under/over-predicted certain points:
residuals = y_test - y_pred

plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Reward")
plt.ylabel("Residual (True - Pred)")
plt.title("Residual Plot")
plt.show()



# --- NOW, DO CLASSIFICATION (via logistic regression as the classifier): ---
# why? question we wanna solve-->: Is the agent in a high (above -2) or low (below -2) combined reward area/range?
df["high_reward"] = (df["combined_reward"] > -2).astype(int)


# now, train the classifier (using log regre):    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df_cls = df.dropna(subset=["high_reward"] + predictors)
Xc = df_cls[predictors]
yc = df_cls["high_reward"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(Xc_train, yc_train)
yc_pred = clf.predict(Xc_test)

acc = accuracy_score(yc_test, yc_pred)
print("Accuracy:", acc)
print("Confusion Matrix:\n", confusion_matrix(yc_test, yc_pred))
print("Report:\n", classification_report(yc_test, yc_pred))
    
