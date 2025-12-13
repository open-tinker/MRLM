"""
gRPC server for hosting environments.

This module provides the EnvironmentServer class that implements the gRPC
EnvironmentService, allowing environments to be accessed remotely.
"""

import logging
import uuid
from concurrent import futures
from typing import Dict, Optional

import grpc

from mrlm.core.base import BaseEnvironment
from mrlm.core.types import Message, Observation, Reward

logger = logging.getLogger(__name__)

# Import protocol buffers (will be available after compilation)
try:
    from mrlm.protocols import mrlm_pb2, mrlm_pb2_grpc
except ImportError:
    logger.warning("Protocol buffers not compiled. Run: python -m mrlm.protocols.compile_proto")
    mrlm_pb2 = None
    mrlm_pb2_grpc = None


class EnvironmentServer:
    """
    gRPC server for hosting environments.

    This server implements the EnvironmentService defined in mrlm.proto.
    It manages multiple environment instances and handles client requests.

    Attributes:
        environments: Dictionary mapping environment IDs to environment instances
        sessions: Dictionary mapping session IDs to (env_id, env) tuples

    Example:
        >>> from mrlm.environments.code import CodeExecutionEnvironment
        >>> envs = {"code": CodeExecutionEnvironment()}
        >>> server = EnvironmentServer(envs)
        >>> # Server is ready to handle gRPC requests
    """

    def __init__(self, environments: Dict[str, BaseEnvironment]):
        """
        Initialize environment server.

        Args:
            environments: Dictionary of environment_id -> environment instance
        """
        self.environments = environments
        self.sessions: Dict[str, tuple[str, BaseEnvironment]] = {}
        logger.info(f"Initialized server with {len(environments)} environments")

    def Reset(self, request, context):
        """Handle environment reset request."""
        if mrlm_pb2 is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Protocol buffers not compiled")
            return None

        env_id = request.environment_id

        if env_id not in self.environments:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Environment '{env_id}' not found")
            return mrlm_pb2.ResetResponse()

        try:
            # Get environment and reset
            env = self.environments[env_id]
            obs = env.reset()

            # Create new session
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = (env_id, env)

            logger.info(f"Reset environment '{env_id}', session '{session_id}'")

            # Convert to proto
            obs_proto = self._observation_to_proto(obs)

            return mrlm_pb2.ResetResponse(
                observation=obs_proto, session_id=session_id, status="success"
            )

        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mrlm_pb2.ResetResponse()

    def Step(self, request, context):
        """Handle environment step request."""
        if mrlm_pb2 is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Protocol buffers not compiled")
            return None

        session_id = request.session_id

        if session_id not in self.sessions:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Session '{session_id}' not found")
            return mrlm_pb2.StepResponse()

        try:
            # Get environment from session
            env_id, env = self.sessions[session_id]

            # Convert action from proto
            action = self._proto_to_message(request.action)

            # Execute step
            obs, reward = env.step(action)

            # Convert to proto
            obs_proto = self._observation_to_proto(obs)
            reward_proto = self._reward_to_proto(reward)

            return mrlm_pb2.StepResponse(
                observation=obs_proto, reward=reward_proto, status="success"
            )

        except Exception as e:
            logger.error(f"Error stepping environment: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return mrlm_pb2.StepResponse()

    def BatchStep(self, request, context):
        """Handle batch step request."""
        if mrlm_pb2 is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Protocol buffers not compiled")
            return None

        responses = []
        for step_request in request.requests:
            response = self.Step(step_request, context)
            responses.append(response)

        return mrlm_pb2.BatchStepResponse(responses=responses)

    def StreamStep(self, request_iterator, context):
        """Handle streaming step requests."""
        for request in request_iterator:
            response = self.Step(request, context)
            yield response

    def ListEnvironments(self, request, context):
        """List available environments."""
        if mrlm_pb2 is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Protocol buffers not compiled")
            return None

        env_infos = []
        for env_id, env in self.environments.items():
            # Filter by type if requested
            if request.environment_type and hasattr(env, "environment_type"):
                if env.environment_type != request.environment_type:
                    continue

            env_info = mrlm_pb2.EnvironmentInfo(
                environment_id=env_id,
                environment_type=getattr(env, "environment_type", "unknown"),
                mode=env.mode.value,
                metadata={},
            )
            env_infos.append(env_info)

        return mrlm_pb2.ListEnvironmentsResponse(environments=env_infos)

    def HealthCheck(self, request, context):
        """Health check endpoint."""
        if mrlm_pb2 is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Protocol buffers not compiled")
            return None

        import time

        return mrlm_pb2.HealthCheckResponse(
            status="healthy",
            info={
                "num_environments": str(len(self.environments)),
                "num_sessions": str(len(self.sessions)),
            },
            timestamp=int(time.time()),
        )

    def _observation_to_proto(self, obs: Observation):
        """Convert Observation to protobuf."""
        messages_proto = [self._message_to_proto(msg) for msg in obs.messages]

        state = {k: str(v) for k, v in (obs.state or {}).items()}
        info = {k: str(v) for k, v in obs.info.items()}

        return mrlm_pb2.ObservationProto(messages=messages_proto, state=state, done=obs.done, info=info)

    def _reward_to_proto(self, reward: Reward):
        """Convert Reward to protobuf."""
        info = {k: str(v) for k, v in reward.info.items()}

        return mrlm_pb2.RewardProto(value=reward.value, components=reward.components, info=info)

    def _message_to_proto(self, msg: Message):
        """Convert Message to protobuf."""
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        metadata = {k: str(v) for k, v in msg.metadata.items()}

        return mrlm_pb2.MessageProto(role=role, content=msg.content, metadata=metadata)

    def _proto_to_message(self, proto) -> Message:
        """Convert protobuf to Message."""
        return Message(role=proto.role, content=proto.content, metadata=dict(proto.metadata))


def serve(
    environments: Dict[str, BaseEnvironment],
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 10,
    max_message_length: int = 100 * 1024 * 1024,  # 100MB
) -> grpc.Server:
    """
    Start gRPC server for environments.

    Args:
        environments: Dictionary of environment_id -> environment
        host: Host to bind to
        port: Port to bind to
        max_workers: Maximum number of worker threads
        max_message_length: Maximum message length in bytes

    Returns:
        Started gRPC server

    Example:
        >>> envs = {"code": CodeExecutionEnvironment()}
        >>> server = serve(envs, port=50051)
        >>> # Server is now running
        >>> server.wait_for_termination()
    """
    if mrlm_pb2_grpc is None:
        raise ImportError(
            "Protocol buffers not compiled. Run: python -m mrlm.protocols.compile_proto"
        )

    # Create server with options
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
        ],
    )

    # Add service
    mrlm_pb2_grpc.add_EnvironmentServiceServicer_to_server(
        EnvironmentServer(environments), server
    )

    # Bind to port
    server_address = f"{host}:{port}"
    server.add_insecure_port(server_address)

    # Start server
    server.start()
    logger.info(f"Server started on {server_address}")
    print(f"✓ Environment server listening on {server_address}")

    return server
