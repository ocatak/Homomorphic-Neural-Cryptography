from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey, EllipticCurvePublicKey
from typing import Tuple
import numpy as np
from numpy.typing import NDArray


def set_curve(curve_name: str) -> ec.EllipticCurve:
    """Sets the elliptic curve based on the given name.

    Args:
        curve_name: The name of the curve to set.

    Returns:
        An instance of an elliptic curve class.

    Raises:
        ValueError: If an invalid curve name is provided.
    """
    if curve_name == "secp224r1":
        return ec.SECP224R1()
    elif curve_name == "secp256k1":
        return ec.SECP256K1()
    elif curve_name == "secp256r1":
        return ec.SECP256R1()
    elif curve_name == "secp384r1":
        return ec.SECP384R1()
    elif curve_name == "secp521r1":
        return ec.SECP521R1()
    else:
        raise ValueError("Invalid curve name")


def get_key_shape(curve: ec.EllipticCurve) -> Tuple[int, int]:
    """Gets the public key and private key shape.

    Args:
        curve: The elliptic curve.

    Returns:
        A tuple of the private key shape and public key shape.
    """
    private_key = ec.generate_private_key(
            curve, default_backend())
    public_key = private_key.public_key()
    pr, pu = convert_key_to_pem(private_key, public_key)
    return pr.size, pu.size


def generate_key_pair(batch_size: int, curve: ec.EllipticCurve) -> Tuple[NDArray[np.object_], NDArray[np.object_]]:
    """Generates a batch of private and public keys.
    
    Args:
        batch_size: The number of keys to generate.
        curve: The elliptic curve.
        
    Returns:
        A tuple of private keys and public keys, each a numpy array of numpy arrays containing float64 elements.
    """
    size = get_key_shape(curve)
    pr_arr = np.empty((batch_size, size[0]))
    pu_arr = np.empty((batch_size, size[1]))
    for i in range(batch_size):
        private_key = ec.generate_private_key(
            curve, default_backend())
        # Derive the associated public key
        public_key = private_key.public_key()
        pr_arr[i], pu_arr[i] = convert_key_to_pem(private_key, public_key)
    return pr_arr, pu_arr


def convert_key_to_pem(private_key: EllipticCurvePrivateKey, public_key: EllipticCurvePublicKey) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert keys to PEM format.

    Args:
        private_key: The private key.
        public_key: The public key.
    
    Returns:
        A tuple containing the bit representations of the private and public keys as numpy arrays of numpy arrays containing float64 elements.
    """
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ).decode()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    return convert_key_to_bit(private_pem), convert_key_to_bit(public_pem)


def convert_key_to_bit(pem: str) -> NDArray[np.int64]:
    """ Converts a PEM-encoded key string to its bit representation as a numpy array.
    
    Args:
        pem: The PEM-encoded key string.
    
    Returns:
        A numpy array of int64 elements representing the key in bits.
    """
    # Convert PEM string to a bit string
    bits = ''.join([format(ord(c), '08b') for c in pem])
    arr = np.array([int(bit) for bit in bits])
    return arr

if __name__ == "__main__":
    curve = set_curve("secp256r1")
    batch_size = 1
    private_key, public_key = generate_key_pair(batch_size, curve)
    np.save(f"key/private_key-{curve.name}-{batch_size}.npy", private_key)
    np.save(f"key/public_key-{curve.name}-{batch_size}.npy", public_key)