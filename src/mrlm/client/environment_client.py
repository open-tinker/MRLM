"""
gRPC client for remote environments.

This module provides the RemoteEnvironment class that wraps a gRPC client
and allows remote environments to be used as if they were local.
"""

import logging
from typing import Optional, Tuple

import grpc

from mrlm.core.base import BaseEnvironment
from mrlm.core.types import EnvironmentMode, Message, Observation, Reward

logger = logging.getLogger(__name__)

# Import protocol buffers (will be available after compilation)
try:
    from mrlm.protocols import mrlm_pb2, mrlm_pb2_grpc
except ImportError:
    logger.warning("Protocol buffers not compiled. Run: python -m mrlm.protocols.compile_proto")
    mrlm_pb2 = None
    mrlm_pb2_grpc = None


class RemoteEnvironment(BaseEnvironment):
    """
    Client wrapper for remote environment accessed via gRPC.

    This class implements the BaseEnvironment interface but forwards all
    calls to a remote environment server via gRPC.

    Attributes:
        environment_id: ID of the environment on the server
        server_address: Address of the gRPC server (host:port)
        session_id: Current session ID (set after reset)

    Example:
        >>> env = RemoteEnvironment(
        ...     environment_id="code",
        ...     server_address="localhost:50051"
        ... )
        >>> obs = env.reset()
        >>> action = Message(role=MessageRole.USER, content="Hello")
        >>> obs, reward = env.step(action)
    """

    def __init__(
        self,
        environment_id: str,
        server_address: str,
        mode: EnvironmentMode = EnvironmentMode.CLIENT,
        max_message_length: int = 100 * 1024 * 1024,  # 100MB
        timeout: float = 60.0,
    ):
        """
        Initialize remote environment client.

        Args:
            environment_id: ID of environment on server
            server_address: Server address (host:port)
            mode: Environment mode (typically CLIENT for remote)
            max_message_length: Maximum message size
            timeout: RPC timeout in seconds
        """
        super().__init__(mode=mode)

        if mrlm_pb2 is None or mrlm_pb2_grpc is None:
            raise ImportError(
                "Protocol buffers not compiled. Run: python -m mrlm.protocols.compile_proto"
            )

        self.environment_id = environment_id
        self.server_address = server_address
        self.session_id: Optional[str] = None
        self.timeout = timeout

        # Create gRPC channel with options
        channel_options = [
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
        ]

        self.channel = grpc.insecure_channel(server_address, options=channel_options)
        self.stub = mrlm_pb2_grpc.EnvironmentServiceStub(self.channel)

        logger.info(f"Connected to remote environment '{environment_id}' at {server_address}")

    def reset(self) -> Observation:
        """
        Reset remote environment.

        Sends a reset request to the server and receives initial observation.

        Returns:
            Initial observation from environment

        Raises:
            grpc.RpcError: If communication fails
        """
        request = mrlm_pb2.ResetRequest(environment_id=self.environment_id)

        try:
            response = self.stub.Reset(request, timeout=self.timeout)

            # Store session ID
            self.session_id = response.session_id

            # Convert observation from proto
            obs = self._proto_to_observation(response.observation)

            logger.info(f"Reset environment, session: {self.session_id}")
            return obs

        except grpc.RpcError as e:
            logger.error(f"Error resetting environment: {e}")
            raise

    def step(self, action: Message) -> Tuple[Observation, Reward]:
        """
        Step remote environment.

        Sends action to server and receives observation and reward.

        Args:
            action: Action message to send

        Returns:
            Tuple of (observation, reward)

        Raises:
            RuntimeError: If environment not reset
            grpc.RpcError: If communication fails
        """
        if self.session_id is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        # Convert action to proto
        action_proto = self._message_to_proto(action)

        request = mrlm_pb2.StepRequest(session_id=self.session_id, action=action_proto)

        try:
            response = self.stub.Step(request, timeout=self.timeout)

            # Convert from proto
            obs = self._proto_to_observation(response.observation)
            reward = self._proto_to_reward(response.reward)

            return obs, reward

        except grpc.RpcError as e:
            logger.error(f"Error stepping environment: {e}")
            raise

    def close(self):
        """Close gRPC channel."""
        if self.channel:
            self.channel.close()
            logger.info(f"Closed connection to {self.server_address}")

    def check_health(self) -> bool:
        """
        Check if server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            request = mrlm_pb2.HealthCheckRequest()
            response = self.stub.HealthCheck(request, timeout=5.0)
            return response.status == "healthy"
        except grpc.RpcError:
            return False

    def _message_to_proto(self, msg: Message):
        """Convert Message to protobuf."""
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        metadata = {k: str(v) for k, v in msg.metadata.items()}

        return mrlm_pb2.MessageProto(role=role, content=msg.content, metadata=metadata)

    def _proto_to_message(self, proto) -> Message:
        """Convert protobuf to Message."""
        return Message(role=proto.role, content=proto.content, metadata=dict(proto.metadata))

    def _proto_to_observation(self, proto) -> Observation:
        """Convert protobuf to Observation."""
        messages = [self._proto_to_message(msg) for msg in proto.messages]
        state = dict(proto.state) if proto.state else None
        info = dict(proto.info) if proto.info else {}

        return Observation(messages=messages, state=state, done=proto.done, info=info)

    def _proto_to_reward(self, proto) -> Reward:
        """Convert protobuf to Reward."""
        components = dict(proto.components) if proto.components else {}
        info = dict(proto.info) if proto.info else {}

        return Reward(value=proto.value, components=components, info=info)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RemoteEnvironment(id='{self.environment_id}', "
            f"server='{self.server_address}', "
            f"session={self.session_id})"
        )
