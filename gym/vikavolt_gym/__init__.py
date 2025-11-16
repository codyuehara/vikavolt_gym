from gym.envs.registration import register
register(
	id='vikavolt-v0',
	entry_point='vikavolt_gym.envs:VikavoltEnv',
	)
