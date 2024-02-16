def generate_test_set(c1_batch, c2_batch):
    """
    Generate test dataset from ciphertext 1 and 2 derived by Alice

    Parameters:
    ----------   
        c1_batch: Ciphertext 1 as numpy array of bits decrypted by Alice using plaintext 1 and the public key
        c2_batch: Ciphertext 2 as numpy array of bits decrypted by Alice using plaintext 2 and the public key

    Returns:
    -------
        X_test: Array containing ciphertext 1 and 2 as numbers
        y_test: Sum of chipertext 1 and 2 as numbers
    """
    X_test = []
    y_test = []
    for i in range(len(c1_batch)):
        p1_binary_string = ''.join(str(bit) for bit in c1_batch[i])
        p2_binary_string = ''.join(str(bit) for bit in c2_batch[i])
        
        X_test.append([ int(p1_binary_string, 2), int(p2_binary_string, 2)])
        y_test.append([ int(p1_binary_string, 2) + int(p2_binary_string, 2)])
    return X_test, y_test

def convert_number_to_bit(predicted):
    """
    Converts the predicted list of numbers to a numpy array of bits, ciphertext 3

    Parameters:
    ----------
        predicted: List of predicted numbers by Homomorphic operation network
    
    Returns:
    -------
        c3_batch: Ciphertext 3. List of predicted numbers converted to bits.
    """
    c3_batch = [[int(bit) for bit in bin(round(number))[2:].zfill(8)] 
            for list_number in predicted 
            for number in list_number]
    return c3_batch