from roveranalyzer.dockerrunner.dockerrunner import (
	DockerCleanup,
	DockerReuse,
	DockerRunner,
)


class ControllerRunner(DockerRunner):
	def __init__(
			self,
			image="sam-dev.cs.hm.edu:5023/rover/crownet/flowcontrol",
			tag="latest",
			docker_client=None,
			name="",
			cleanup_policy=DockerCleanup.REMOVE,
			reuse_policy=DockerReuse.REMOVE_STOPPED,
			detach=False,
			journal_tag="",
	):
		super().__init__(
			image,
			tag,
			docker_client=docker_client,
			name=name,
			cleanup_policy=cleanup_policy,
			reuse_policy=reuse_policy,
			detach=detach,
			journal_tag=journal_tag,
		)
