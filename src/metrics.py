import tensorflow as tf

def kl_diversity_log(mu_i, log_sigma_i, mu_j, log_sigma_j, dimension):
    '''calculate the kl diversity of two normal distribution.
    support each dimension in sigma is independent, store the diagonal log value of variance matrix in log_var as vector
    mu: [batch size, h_dimension]
    sigma: [batch size, h_dimension]
    '''
    sigma_i = tf.exp(log_sigma_i)
    sigma_j = tf.exp(log_sigma_j)
    sigma_ratio = tf.exp(log_sigma_i - log_sigma_j)
    term1 = tf.reduce_sum(sigma_ratio, axis=1)
    log_term = tf.reduce_sum(log_sigma_i - log_sigma_j, axis=1)
    #log_term = tf.reduce_sum(tf.log(sigma_i+1e-14), axis=1) - tf.reduce_sum(tf.log(sigma_j+1e-14), axis=1)
    mu = mu_i - mu_j
    term2 = tf.reduce_sum(tf.div(tf.square(mu), sigma_j), axis=1)
    kl = 0.5 * (term1 + term2 - log_term - dimension)
    return kl




def kl_diversity(mu_i, sigma_i, mu_j, sigma_j, dimension):
    '''calculate the kl diversity of two normal distribution.
    support each dimension in sigma is independent, store the diagonal value of variance matrix in sigma as vector
    mu: [batch size, h_dimension]
    sigma: [batch size, h_dimension]
    '''
    #distribution_i = tf.distributions.Normal(loc=mu_i, scale=sigma_i)
    #distribution_j = tf.distributions.Normal(loc=mu_j, scale=sigma_j)

    #kl = tf.distributions.kl_divergence(distribution_i, distribution_j)

#self implemented kl diversity, meet nan problem because div 0
    sigma_ratio = tf.div(sigma_i, sigma_j)
    term1 = tf.reduce_sum(sigma_ratio, axis=1)
    log_term = tf.reduce_sum(tf.log(sigma_ratio+1e-14), axis=1)
    #log_term = tf.reduce_sum(tf.log(sigma_i+1e-14), axis=1) - tf.reduce_sum(tf.log(sigma_j+1e-14), axis=1)
    mu = mu_i - mu_j
    term2 = tf.reduce_sum(tf.div(tf.square(mu), sigma_j), axis=1)
    kl = 0.5 * (term1 + term2 - log_term - dimension)
    return kl

def js_diversity(mu_i, sigma_i, mu_j, sigma_j, dimension):
    '''calculate the js diversity of two normal distribution.
    support each dimension in sigma is independent, store the diagonal value of variance matrix in sigma as vector
    mu: [batch size, h_dimension]
    sigma: [batch size, h_dimension]
    '''

    mu = (mu_i + mu_j) / 2
    sigma = (sigma_i + sigma_j) / 2
    js = kl_diversity(mu_i, sigma_i, mu, sigma, dimension) + kl_diversity(mu_j, sigma_j, mu, sigma, dimension)
    return js

def square_exponential_loss(postive_energy, negative_energy, neg_weight=1.0):
    """
    square_exponential_loss give low energy to postive pair and high energy to negative pair.
    These loss does not have a fixed margin(e.g. hinge loss),
    and pushes the energy of negative term to infinity with exponentially decreasing force.
    """
    #loss = tf.reduce_mean(postive_energy) + 0. * tf.reduce_mean(tf.exp(-negative_energy))
    loss = tf.reduce_mean(postive_energy) + neg_weight * tf.reduce_mean(tf.exp(-negative_energy))
    #loss = tf.reduce_mean(tf.square(postive_energy)) + neg_weight * tf.reduce_mean(tf.exp(-negative_energy))
    #loss = tf.reduce_mean(loss)
    return loss


