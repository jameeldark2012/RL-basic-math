# %% [markdown]
# # 1.RL Sample Average Method

# %%
import torch   
import random
torch.manual_seed(100)



# %%
if torch.cuda.is_available():
    device = "cuda"

device

# %%
BATCH_SIZE = 100
BATCH_SIZE

# %%
actions = torch.randint(1 , 6 , (BATCH_SIZE,) , device=device)
actions

# %%
actions.shape

# %%
rewards = torch.rand((BATCH_SIZE,) , device=device)
rewards

# %%
rewards.shape

# %%
actions_rewards = torch.stack(  (actions , rewards) , dim=1)
actions_rewards

# %%
actions_rewards[actions_rewards[: , 0] == 5.0]

# %%
mean_rewards = torch.stack([rewards[actions == i].mean() for i in actions.unique()])
mean_rewards

# %% [markdown]
# ### 1.1 Increamental Update Rule for non stationary enviroments : <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><msub><mi>Q</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>Q</mi><mi>n</mi></msub><mo>+</mo><mi>α</mi><mo stretchy="false">(</mo><msub><mi>R</mi><mi>n</mi></msub><mo>−</mo><msub><mi>Q</mi><mi>n</mi></msub><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">Q_{n+1} = Q_n + \alpha (R_n - Q_n) 
# </annotation></semantics></math>
# 
# ( Q_n ) is the current estimate of the action value.
# ( R_n ) is the reward received after taking the action.
# ( alpha ) is the step size parameter, which determines how much weight to give to the new reward compared to the old estimate2.
# 

# %%
def update(qn , a , rn ):

    try:
        for i in range(len(rn)):
            qn = qn + a * (rn[i] - qn)
            print(qn)
        return qn

    except:
        qn_1 = qn + a * (rn - qn)
        return qn_1

    

rn = torch.tensor(0.9 , device=device )
qn = mean_rewards[0]
a = torch.tensor(1/len(actions_rewards) , device= device)

# %%
qn

# %%
qn_1 = update(qn ,a , rn)
qn_1

# %% [markdown]
# ### Experiement 

# %%
qn = torch.tensor(5 , device=device)
a = torch.tensor(0.5 , device=device)
rn = torch.randint(0 , 2 , (100,) , device=device)
rn

# %%
#### The q0 value got updated based on our rewards , missing aroudn with qn initial value affects the process naking the agent more optimistic or pessemstic 

# %%
qn = update(qn ,a , rn)
qn

# %% [markdown]
# ### Epsilon Greedy Agent Example
# 
# <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><msub><mi>Q</mi><mrow><mi>n</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>Q</mi><mi>n</mi></msub><mo>+</mo><mi>α</mi><mo stretchy="false">(</mo><msub><mi>R</mi><mi>n</mi></msub><mo>−</mo><msub><mi>Q</mi><mi>n</mi></msub><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">Q_{n+1} = Q_n + \alpha (R_n - Q_n) 
# </annotation></semantics></math>

# %% [markdown]
# #### Training

# %%
actions = torch.tensor([0 ,1] , device= device )
Q = torch.tensor([5, 5] , device=device , dtype=float)
BATCH_SIZE = 1000
EPS = 0.1
a = 0.5
probabilities = torch.tensor([0.2, 0.8], device=device)  # 70% for action 0, 30% for action 1

# Sample actions based on the defined probabilities
actions_samples = torch.multinomial(probabilities, BATCH_SIZE, replacement=True )
actions_samples

# %%
rewards_samples = torch.multinomial(probabilities, BATCH_SIZE, replacement=True )
rewards_samples = torch.tensor(rewards_samples , device=device ,dtype=torch.float)
rewards_samples




# %%

for i in range(BATCH_SIZE):

    if random.random() > EPS :

        reward = rewards_samples[i]
        action = actions_samples[i]
        Q[int(action)] = update(Q[int(action)] ,a , reward)

    else :
        action = torch.argmax(Q , dim=0)

        if action == actions_samples[i]:
            reward = rewards_samples[i]
            Q[int(action)] = update(Q[int(action)] ,a , reward)
        
        else:
            reward = torch.tensor(0 , device=device , dtype=torch.float) 
            Q[int(action)] = update(Q[int(action)] ,a , reward)
    
    print(Q[0].item() , Q[1].item() , reward.item() ," Action : " ,action.item())
        


# %% [markdown]
# #### Evaluation

# %%
def evaluate_agent():

        EVAL_TIME_STEPS = 100
        TOTAL_REWARD = 0
        N_EPISODES = 10
        action = torch.argmax(Q , dim=0)
        print("Chosen action is" , int(action))

        for i in range(N_EPISODES):
                EPISODE_REWARD = 0
                for i in range(EVAL_TIME_STEPS):

                        action = torch.argmax(Q , dim=0)
                        random_index_sample = random.randint(0 , len(actions_samples) - 1)

                        if action == actions_samples[random_index_sample]:
                                reward = rewards_samples[random_index_sample]
                        else :
                                reward = torch.tensor(0 , device=device , dtype=torch.float)
                        
                        EPISODE_REWARD+=reward
                print("Episode Reward" , EPISODE_REWARD)

                TOTAL_REWARD+=EPISODE_REWARD



        print("Mean Reward: ", (TOTAL_REWARD / N_EPISODES).item()  )



# %%
evaluate_agent()

# %% [markdown]
# ### Upper confidence bound

# %% [markdown]
# ### <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><msub><mi>A</mi><mi>t</mi></msub><mo>=</mo><mi>arg</mi><mo>⁡</mo><munder><mrow><mi>max</mi><mo>⁡</mo></mrow><mi>a</mi></munder><mrow><mo fence="true">(</mo><msub><mi>Q</mi><mi>t</mi></msub><mo stretchy="false">(</mo><mi>a</mi><mo stretchy="false">)</mo><mo>+</mo><mi>c</mi><msqrt><mfrac><mrow><mi>ln</mi><mo>⁡</mo><mi>t</mi></mrow><mrow><msub><mi>N</mi><mi>t</mi></msub><mo stretchy="false">(</mo><mi>a</mi><mo stretchy="false">)</mo></mrow></mfrac></msqrt><mo fence="true">)</mo></mrow></mrow><annotation encoding="application/x-tex">A_t = \arg\max_a \left( Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right)
# </annotation></semantics></math>

# %% [markdown]
# <math xmlns=“http://www.w3.org/1998/Math/MathML” display=“inline”><semantics><mrow><msub><mi>Q</mi><mi>t</mi></msub><mo stretchy=“false”>(</mo><mi>a</mi><mo stretchy=“false”>)</mo></mrow><annotation encoding=“application/x-tex”>Q_t(a)</annotation></semantics></math>: This is the estimated value of action ( a ) at time ( t ). It represents the average reward received from action ( a ) up to time ( t ).
# <math xmlns=“http://www.w3.org/1998/Math/MathML” display=“inline”><semantics><mrow><mi>c</mi></mrow><annotation encoding=“application/x-tex”>c</annotation></semantics></math>: This is a constant that controls the degree of exploration. A higher value of ( c ) encourages more exploration.
# <math xmlns=“http://www.w3.org/1998/Math/MathML” display=“inline”><semantics><mrow><msqrt><mfrac><mrow><mi>ln</mi><mo>⁡</mo><mi>t</mi></mrow><mrow><msub><mi>N</mi><mi>t</mi></msub><mo stretchy=“false”>(</mo><mi>a</mi><mo stretchy=“false”>)</mo></mrow></mfrac></msqrt></mrow><annotation encoding=“application/x-tex”>\sqrt{\frac{\ln t}{N_t(a)}}</annotation></semantics></math>: This is the exploration term. It increases as the number of times action ( a ) has been selected, ( N_t(a) ), decreases. It also increases with the logarithm of the current time step ( t ), encouraging exploration of less frequently chosen actions.
# <math xmlns=“http://www.w3.org/1998/Math/MathML” display=“inline”><semantics><mrow><mi>arg</mi><mo>⁡</mo><munder><mrow><mi>max</mi><mo>⁡</mo></mrow><mi>a</mi></munder></mrow><annotation encoding=“application/x-tex”>\arg\max_a</annotation></semantics></math>: This notation means that we select the action ( a ) that maximizes the entire expression.

# %%
c = 2
Q = torch.tensor([5, 5] , device=device , dtype=float)
i =0

actions_freq = torch.ones((2,) , device=device , dtype=torch.float)

def upper_confidence_bound(Q , c , t , nt_a):

    for i in range(len(Q)):
          
          Q[i] = Q[i] + c * (   torch.sqrt( torch.log(t)  ) /  nt_a[i]      )
    return Q

for i in range(BATCH_SIZE):

    
    action = torch.argmax( upper_confidence_bound(Q , c ,torch.tensor(i+1 , device=device) , actions_freq) , dim=0)

    if action == actions[0]:

        actions_freq[0]+= torch.tensor(1 , device=device , dtype=torch.float)
    else :
        actions_freq[1]+= torch.tensor(1 , device=device , dtype=torch.float)       

    if action == actions_samples[i]:
            reward = rewards_samples[i]
            Q[int(action)] = update(Q[int(action)] ,a , reward)
        
    else:
            reward = torch.tensor(0 , device=device , dtype=torch.float) 
            Q[int(action)] = update(Q[int(action)] ,a , reward)
    
    print(Q[0].item() , Q[1].item() , reward.item() ," Action : " ,action.item())

# %%
evaluate_agent()

# %% [markdown]
# # Decaying Past Rewards
# 
# $$
# Q = (1 - \alpha)^n Q_1 + \sum_{i=1}^{n} \alpha(1 - \alpha)^{n-i} R_i
# $$
# 

# %% [markdown]
# # Expected returns
#  The Expected retruns over the episode is used for the agent to maximize, the expected concept is used since the state rewards must be represented in probabilities and are not certain
#  # <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi mathvariant="double-struck">E</mi><mo stretchy="false">[</mo><mi>R</mi><mo stretchy="false">]</mo><mo>=</mo><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msub><mi>p</mi><mi>i</mi></msub><mo>⋅</mo><msub><mi>r</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">\mathbb{E}[R] = \sum_{i=1}^{n} p_i \cdot r_i
# </annotation></semantics></math>
# 
# # <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><mi mathvariant="double-struck">E</mi><mo stretchy="false">[</mo><mi>R</mi><mo stretchy="false">]</mo><mo>=</mo><msubsup><mo>∫</mo><mi>a</mi><mi>b</mi></msubsup><mi>r</mi><mo>⋅</mo><mi>p</mi><mo stretchy="false">(</mo><mi>r</mi><mo stretchy="false">)</mo><mtext> </mtext><mi>d</mi><mi>r</mi></mrow><annotation encoding="application/x-tex">\mathbb{E}[R] = \int_{a}^{b} r \cdot p(r) \, dr
# </annotation></semantics></math>

# %% [markdown]
# # Reward Recursive Equation
# 
# # <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><msub><mi>G</mi><mi>t</mi></msub><mo>=</mo><msub><mi>R</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>+</mo><mi>γ</mi><msub><mi>G</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub></mrow><annotation encoding="application/x-tex">G_t = R_{t+1} + \gamma G_{t+1}
# </annotation></semantics></math>
# 
# <math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mtable rowspacing="0.25em" columnalign="right left" columnspacing="0em"><mtr><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><mo stretchy="false">(</mo><msub><mi>G</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><mrow></mrow><mo>:</mo><mtext>Expected&nbsp;return&nbsp;at&nbsp;time&nbsp;</mtext><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mi mathvariant="normal">.</mi></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><mo stretchy="false">(</mo><msub><mi>R</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo stretchy="false">)</mo></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><mrow></mrow><mo>:</mo><mtext>Reward&nbsp;received&nbsp;at&nbsp;time&nbsp;</mtext><mo stretchy="false">(</mo><mi>t</mi><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo><mi mathvariant="normal">.</mi></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><mo stretchy="false">(</mo><mi>γ</mi><mo stretchy="false">)</mo></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><mrow></mrow><mo>:</mo><mtext>Discount&nbsp;factor,&nbsp;</mtext><mo stretchy="false">(</mo><mn>0</mn><mo>≤</mo><mi>γ</mi><mo>≤</mo><mn>1</mn><mo stretchy="false">)</mo><mi mathvariant="normal">.</mi></mrow></mstyle></mtd></mtr><mtr><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><mo stretchy="false">(</mo><msub><mi>G</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo stretchy="false">)</mo></mrow></mstyle></mtd><mtd><mstyle scriptlevel="0" displaystyle="true"><mrow><mrow></mrow><mo>:</mo><mtext>Expected&nbsp;return&nbsp;at&nbsp;time&nbsp;</mtext><mo stretchy="false">(</mo><mi>t</mi><mo>+</mo><mn>1</mn><mo stretchy="false">)</mo><mi mathvariant="normal">.</mi></mrow></mstyle></mtd></mtr></mtable><annotation encoding="application/x-tex">\begin{align*}
# ( G_t ) &amp;: \text{Expected return at time } ( t ). \\
# ( R_{t+1} ) &amp;: \text{Reward received at time } ( t+1 ). \\
# ( \gamma ) &amp;: \text{Discount factor, } ( 0 \leq \gamma \leq 1 ). \\
# ( G_{t+1} ) &amp;: \text{Expected return at time } ( t+1 ).
# \end{align*}
# </annotation></semantics></math>

# %% [markdown]
# #### The second term is the function calling itself on time step t+1

# %%
rewards = torch.tensor([1,1,1,1,1] , device=device)
y = torch.tensor(0.8 , device=device)

def calculate_reward(rt , y):

    gt = 0
    t = 0
    
    for reward in rt:
         gt+= y **t * reward
         t+=1
    
    return gt

 
    
gt = calculate_reward(rewards , y )

print(gt)


