from typing import List, Dict, Callable, Iterable, Optional
import torch
# import torch_pso

# Adapting base PSO optimizer to account for different sizes/magnitudes of our parameters in SAL
class ParticleDifferentScales(torch_pso.optim.ParticleSwarmOptimizer.Particle):
    r'''
    Updates from Particle class to enable different scales for different parameters. Keeps same .step() method
    '''
    def __init__(self,
                 param_groups: List[Dict],
                 inertial_weight: float,
                 cognitive_coefficient: float,
                 social_coefficient: float,
                 max_param_values: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 min_param_values: list = [-1.0, -1.0, -1.0, 0, 0, -1.0, -1.0]):

        self.param_groups = param_groups
        self.position = [{'params': [(h-l)*torch.rand(()) + l for h, l in zip(max_param_values, min_param_values)]}]        
        self.velocity = [{'params': [(h-l)*torch.rand(()) + l for h, l in zip(max_param_values, min_param_values)]}]
        self.best_known_position = torch_pso.optim.GenericPSO.clone_param_groups(self.position)
        self.best_known_loss_value = torch.inf

        self.inertial_weight = inertial_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient

        # Parameters for clipping values within boundaries
        self.max_param_values = max_param_values
        self.min_param_values = min_param_values
    
    def clip_position(self, position, high, low):
        '''Ensures that parameters are kept within the boundaries defined at the experiment start.'''
        position = torch.tensor([position, high]).min() # clip if too high
        position = torch.tensor([position, low]).max() # clip if too low
        return position

    def step(self, closure: Callable[[], torch.Tensor], global_best_param_groups: List[Dict]) -> torch.Tensor:
        """
        Particle will take one step.
        :param closure: A callable that reevaluates the model and returns the loss.
        :param global_best_param_groups: List of param_groups that yield the best found loss globally
        :return:
        """
        # Because our parameters are not a single tensor, we have to iterate over each group, and then each param in
        # each group.
        for position_group, velocity_group, personal_best, global_best, master in zip(self.position, self.velocity,
                                                                                    self.best_known_position,
                                                                                    global_best_param_groups,
                                                                                    self.param_groups):
            position_group_params = position_group['params']
            velocity_group_params = velocity_group['params']
            personal_best_params = personal_best['params']
            global_best_params = global_best['params']
            master_params = master['params']

            new_position_params = []
            new_velocity_params = []
            for p, v, pb, gb, m, h, l in zip(position_group_params, velocity_group_params, personal_best_params,
                                    global_best_params, master_params, self.max_param_values, self.min_param_values):
                rand_personal = torch.rand_like(v)
                rand_group = torch.rand_like(v)
                new_velocity = (self.inertial_weight * v
                                + self.cognitive_coefficient * rand_personal * (pb - p)
                                + self.social_coefficient * rand_group * (gb - p)
                                )
                new_velocity_params.append(new_velocity)
                new_position = self.clip_position(p + new_velocity, high=h, low=l)
                new_position_params.append(new_position)
                m.data = new_position.data  # Update the model, so we can use it for calculating loss
            position_group['params'] = new_position_params
            velocity_group['params'] = new_velocity_params

        # Really crummy way to update the parameter weights in the original model.
        # Simply changing self.param_groups doesn't update the model.
        # Nor does changing its elements or the raw values of 'param' of the elements.
        # We have to change the underlying tensor data to point to the new positions
        for i in range(len(self.position)):
            for j in range(len(self.param_groups[i]['params'])):
                self.param_groups[i]['params'][j].data = self.param_groups[i]['params'][j].data

        # Calculate new loss after moving and update the best known position if we're in a better spot
        new_loss = closure()
        if new_loss < self.best_known_loss_value:
            self.best_known_position = torch_pso.optim.GenericPSO.clone_param_groups(self.position)
            self.best_known_loss_value = new_loss
        return new_loss
    

class ParticleSwarmOptimizerCustom(torch_pso.GenericPSO):
    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 inertial_weight: float = .9,
                 cognitive_coefficient: float = 1.,
                 social_coefficient: float = 1.,
                 num_particles: int = 100,
                 max_param_values: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 min_param_values: list = [-1.0, -1.0, -1.0, 0, 0, -1.0, -1.0]):
        self.num_particles = num_particles
        self.inertial_weight = inertial_weight
        self.cognitive_coefficient = cognitive_coefficient
        self.social_coefficient = social_coefficient
        self.max_param_values = max_param_values
        self.min_param_values = min_param_values

        kwargs = {'inertial_weight': inertial_weight,
                  'cognitive_coefficient': cognitive_coefficient,
                  'social_coefficient': social_coefficient,
                  'max_param_values': max_param_values,
                  'min_param_values': min_param_values}
        super().__init__(params, num_particles, ParticleDifferentScales, particle_kwargs=kwargs)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor], particle_step_kwargs: Optional[Dict] = None) -> torch.Tensor:
        """
        Performs a single optimization step.

        :param particle_step_kwargs: Dict of keyword arguments to pass to the particle step function, if needed.
        :param closure: A callable that reevaluates the model and returns the loss.
        :return: the final loss after the step (as calculated by the closure)
        """
        if particle_step_kwargs is None:
            particle_step_kwargs = {}
        for idx, particle in enumerate(self.particles):
            particle_loss = particle.step(closure, self.best_known_global_param_groups, **particle_step_kwargs)
            print(f'Particle #{idx}: loss={particle_loss}, params={particle.position[0]["params"]}')
            if particle_loss < self.best_known_global_loss_value:
                self.best_known_global_param_groups = torch_pso.optim.GenericPSO.clone_param_groups(particle.position)
                self.best_known_global_loss_value = particle_loss
        print()
        # set the module's parameters to be the best performing ones
        for master_group, best_group in zip(self.param_groups, self.best_known_global_param_groups):
            clone = torch_pso.optim.GenericPSO.clone_param_group(best_group)['params']
            for i in range(len(clone)):
                master_group['params'][i].data = clone[i].data

        return closure()  # loss = closure()

    subclasses = []