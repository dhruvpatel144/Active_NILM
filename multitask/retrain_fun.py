from sklearn.preprocessing import StandardScaler
from dataloader import *
from utilities import *
import jax.numpy as jnp
import matplotlib.pyplot as plt
from model import *
import datetime


def retrain(new_house, new_start, new_end, pool_houses, train, test, given_date, data_aggregated):
    n = 99
    if new_house == 0:
        x_train, y_train = dataloader(train, "2018-03-01 00:00:00-06", "2018-03-10 23:59:00-06", n)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_train = scaler_x.fit_transform(x_train)
        y_train = scaler_y.fit_transform(y_train)
        x_train = jnp.array(x_train).reshape(x_train.shape[0], n, 1)
        y_train = jnp.array(y_train)
        print(y_train.shape)
        model = seq2point()
        params = model.init(jax.random.PRNGKey(0), x_train, True)
        params, losses = fit(model, params, x_train, y_train, False, batch_size=2048, learning_rate=0.0001, epochs=30)
        plt.plot(losses)
        plt.show()
        x_test, y_test = dataloader(test, "2018-05-01 00:00:00-06", "2018-05-10 23:59:00-06", n)
        x_test = scaler_x.transform(x_test)
        x_test = jnp.array(x_test).reshape(x_test.shape[0], n, 1)
        y_test = np.array(y_test)
        y_hat = model.apply(params, x_test, True, rngs={"dropout": jax.random.PRNGKey(0)})
        print(np.array(y_hat).shape)

        n_stacks = 10
        test_mean = scaler_y.inverse_transform(y_hat[0])
        test_sigma = scaler_y.scale_ * y_hat[1]
        print(
            f"RMSE : {rmse(y_test, test_mean)} MAE  : {mae(y_test,test_mean)} NLL : {NLL(test_mean,test_sigma,y_test)}"
        )

    else:
        new_df = data_aggregated[
            ((data_aggregated["dataid"] == new_house) & (data_aggregated["localminute"] > new_start))
        ]
        train = train.append(new_df)
        print("Train houses are")
        print(train["dataid"].unique())

        x_test, y_test = dataloader(test, "2018-05-01 00:00:00-06", "2018-05-10 23:59:00-06", n)

        end_date = new_end
        x_train, y_train = dataloader(train, "2018-03-01 00:00:00-06", end_date, n)
        # print(x_train.shape)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_train = scaler_x.fit_transform(x_train)
        y_train = scaler_y.fit_transform(y_train)
        x_train = jnp.array(x_train).reshape(x_train.shape[0], n, 1)
        y_train = jnp.array(y_train)
        model = seq2point()
        params = model.init(jax.random.PRNGKey(0), x_train, True)
        params, losses = fit(model, params, x_train, y_train, False, batch_size=2048, learning_rate=0.0001, epochs=30)
        plt.plot(losses)
        plt.show()

        x_test = scaler_x.transform(x_test)
        x_test = jnp.array(x_test).reshape(x_test.shape[0], n, 1)
        y_test = np.array(y_test)
        n_stacks = 10
        y_hat = model.apply(params, x_test, True, rngs={"dropout": jax.random.PRNGKey(0)})
        test_mean = scaler_y.inverse_transform(y_hat[0])
        test_sigma = scaler_y.scale_ * y_hat[1]

        print(
            f"RMSE : {rmse(y_test, test_mean)} MAE  : {mae(y_test,test_mean)} NLL : {NLL(test_mean,test_sigma,y_test)}"
        )

    max_uncertainity = 0
    max_house_id = 0
    max_house = 0

    for i in range(len(pool_houses)):
        pool_data = data_aggregated[data_aggregated["dataid"] == pool_houses[i]]
        date_str = given_date + "00"  # Add a day and fix the format
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S%z")
        date_plus_1_day = date + datetime.timedelta(days=1)
        date_end = str(date_plus_1_day)
        mc_test_sigma_list = []
        for j in range(15):
            x_pool, y_pool = dataloader(pool_data, date_str, date_end, n)
            # print(x_pool.shape)
            x_pool = scaler_x.transform(x_pool)
            x_pool = np.array(x_pool).reshape(x_pool.shape[0], n, 1)

            n_stacks = 10
            fn = lambda x, i: model.apply(params, x, False, rngs={"dropout": jax.random.PRNGKey(i)})
            y_stacks = jax.vmap(jax.jit(fn), in_axes=(None, 0))(x_pool, jnp.arange(n_stacks))
            y_true = scaler_y.inverse_transform(y_test)
            mc_test_mean, mc_test_sigma = gmm_mean_var(y_stacks[0], y_stacks[1])
            mc_test_mean = scaler_y.inverse_transform(mc_test_mean)
            mc_test_sigma = scaler_y.scale_ * mc_test_sigma
            test_sigma_mean = mc_test_sigma.mean(axis=0)
            test_sigma_mean_of_appliances = test_sigma_mean.mean()
            mc_test_sigma_list.append(test_sigma_mean_of_appliances)
        # test_mean = scaler_y.inverse_transform(y_hat[0])
        weights = np.array(
            [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
        )
        mc_test_sigma_weighted = np.array(mc_test_sigma_list) * weights
        mc_test_sigma_weighted = mc_test_sigma_weighted.mean()

        if mc_test_sigma_weighted > max_uncertainity:
            max_uncertainity = mc_test_sigma_weighted
            max_house_id = i
            max_house = pool_houses[i]

    return max_house, max_house_id, rmse(y_test, test_mean), mae(y_test, test_mean), train, test


def retrain_random(new_house, new_start, new_end, train, test, data_aggregated):
    # train = data_aggregated[data_aggregated["dataid"].isin(train_houses)]
    # test = data_aggregated[data_aggregated["dataid"].isin(test_houses)]
    n = 99
    if new_house == 0:

        x_train, y_train = dataloader(train, "2018-03-01 00:00:00-06", "2018-03-10 23:59:00-06", n)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_train = scaler_x.fit_transform(x_train)
        y_train = scaler_y.fit_transform(y_train)
        x_train = jnp.array(x_train).reshape(x_train.shape[0], n, 1)
        y_train = jnp.array(y_train)
        print(y_train.shape)
        model = seq2point()
        params = model.init(jax.random.PRNGKey(0), x_train, True)
        params, losses = fit(model, params, x_train, y_train, False, batch_size=2048, learning_rate=0.0001, epochs=30)
        plt.plot(losses)
        plt.show()
        x_test, y_test = dataloader(test, "2018-05-01 00:00:00-06", "2018-05-10 23:59:00-06", n)
        x_test = scaler_x.transform(x_test)
        x_test = jnp.array(x_test).reshape(x_test.shape[0], n, 1)
        y_test = np.array(y_test)
        y_hat = model.apply(params, x_test, True, rngs={"dropout": jax.random.PRNGKey(0)})
        print(np.array(y_hat).shape)
        n_stacks = 10
        test_mean = scaler_y.inverse_transform(y_hat[0])
        test_sigma = scaler_y.scale_ * y_hat[1]
        current_y_hat = test_mean
        print(
            f"RMSE : {rmse(y_test, test_mean)} MAE  : {mae(y_test,test_mean)} NLL : {NLL(test_mean,test_sigma,y_test)}"
        )

    else:
        new_df = data_aggregated[
            ((data_aggregated["dataid"] == new_house) & (data_aggregated["localminute"] > new_start))
        ]
        train = train.append(new_df)
        print("Train houses are")
        print(train["dataid"].unique())
        x_test, y_test = dataloader(test, "2018-05-01 00:00:00-06", "2018-05-10 23:59:00-06", n)
        end_date = new_end
        x_train, y_train = dataloader(train, "2018-03-01 00:00:00-06", end_date, n)
        # print(x_train.shape)
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_train = scaler_x.fit_transform(x_train)
        y_train = scaler_y.fit_transform(y_train)
        x_train = jnp.array(x_train).reshape(x_train.shape[0], n, 1)
        y_train = jnp.array(y_train)
        model = seq2point()
        params = model.init(jax.random.PRNGKey(0), x_train, True)
        params, losses = fit(model, params, x_train, y_train, False, batch_size=2048, learning_rate=0.0001, epochs=30)
        plt.plot(losses)
        plt.show()
        x_test = scaler_x.transform(x_test)
        x_test = jnp.array(x_test).reshape(x_test.shape[0], n, 1)
        y_test = np.array(y_test)
        n_stacks = 10
        y_hat = model.apply(params, x_test, True, rngs={"dropout": jax.random.PRNGKey(0)})
        test_mean = scaler_y.inverse_transform(y_hat[0])
        test_sigma = scaler_y.scale_ * y_hat[1]
        current_y_hat = test_mean
        print(
            f"RMSE : {rmse(y_test, test_mean)} MAE  : {mae(y_test,test_mean)} NLL : {NLL(test_mean,test_sigma,y_test)}"
        )
    return rmse(y_test, test_mean), mae(y_test, test_mean), train, test, current_y_hat
