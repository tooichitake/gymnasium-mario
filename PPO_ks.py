
Algorithm: PPO with Kickstarting (KS) Regularization

1: Initialize actor πθ, critic Vϕ, and load frozen pretrained policy π*
2: for iteration = 1 to M do
3:   Run policy πθ for T timesteps and collect (st, at, rt)
4:   Compute advantages Ât using GAE
5:   for epoch = 1 to N do
6:     for minibatch B ⊂ {(st, at, Ât)} do
7:       Compute PPO losses:
          L^CLIP(θ) = Êt[min(rt(θ)Ât, clip(rt(θ), 1-ε, 1+ε)Ât)]
          L^VF(ϕ) = Êt[(Vϕ(st) - V̂t)²]
          
8:       Compute KS loss:
          L^KS(θ) = Êt[log π*(at|st) - log πθ(at|st)]
          
9:       Update policy by gradient ascent:
          θ ← θ + α∇θ(L^CLIP(θ) - c₁L^VF(θ) + c₂S[πθ] + βL^KS(θ))
          
10:  end for
11: end for