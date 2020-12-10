import numpy as np
import math
from opt_einsum import contract
import tensornetwork as tn
tn.set_default_backend("pytorch")
import torch
torch.set_default_tensor_type(torch.DoubleTensor)


def d(r):
    """Kronecker delta 3-tensor with specified bond dimension r, contraction direction is up to down and right to left."""

    i = np.arange(r)
    res = torch.zeros(r,r,r)
    res[i, i, i] = 1.

    return tn.Node(res, name="delta")


def contr(L, g1, g2, single=False):
    """Calculates TN contractions between g1^(l)_{ij}, g2^(l)_{kl} and L^(l+1).
        L should be that of the next layer, single is for contacting only the network with a basis element."""

    if single:
        assert g1.tensor.shape[0] == g2.tensor.shape[0]
    else:
        assert g1.tensor.shape[0] == g1.tensor.shape[1] and g2.tensor.shape[0] == g2.tensor.shape[1] and g1.tensor.shape[0] == g2.tensor.shape[0]
   
    # create downstairs copy of L and two delta tensors, if needed
    d1 = d(g1.tensor.shape[0])
    if not single:
        Lcopy = list(tn.copy((L,))[0].values())[0]
        d2 = d(g1.tensor.shape[0])

    # setup links and contract from top to bottom
    L[1] ^ d1[0]
    d1[1] ^ g1[0]
    d1[2] ^ g2[0]
    if not single:
        g1[1] ^ d2[1]
        g2[1] ^ d2[0]
        d2[2] ^ Lcopy[1]

    # perform all contractions and return order-2 tensor
    if single:
        return tn.contractors.greedy((L, d1, g1, g2), output_edge_order=(L[0],))
    else:
        return tn.contractors.greedy((L, d1, g1, g2, d2, Lcopy), output_edge_order=(L[0], Lcopy[0]))


def marginal_contr(net, x=np.array([]), half=False, c0=1./math.sqrt(2), evalmode=True, legendre=False):
    """Fully contracts TN with itself, with possible input inclusion in specified pixels x, giving the unnormalized marginalized probability distribution.
        Outputs order-two tensor (outer legs)."""

    # x should be 1d np array of values for pixels in SEQUENCE (1, ..., x.shape[0])
    assert x.shape[0] >= 0 and x.shape[0] <= 256

    if evalmode:
        net.eval() # sets batchnorm layers of net to evaluation mode
    else:
        net.train()

    # Determine maximum number of iterations for the hierarchical contractions
    if half:
        max_it = 7
    else:
        max_it = 8

    # Build the node tensors L^(l):=L^(l,j) (j -> same at every pixel), with (up to down) indices L^l_{BI}
    temp = [] # First store all parameters in temp, then proceed with the network-specific allocations into L
    ttemp = [] # store running mean and variance of norm layers, or batch mean and variance if eval=False

    if net.device == 'cpu':
        for param in net.parameters():
            # make conv layers into matrices with torch shape (output_channels,input_channels)
            # and batchnorm layers into matrices with torch shape (input_channels, 1)
            temp.append(param.view(param.data.shape[0], -1))
        m = 0
        for module in net.modules():
            if hasattr(module, "running_mean"):
                if evalmode:
                    ttemp.append(module.running_mean) # torch shape (input_channels,)
                    ttemp.append(module.running_var) # torch shape (input_channels,)
                else:
                    ttemp.append(net.batchmean[m]) # torch shape (input_channels,)
                    ttemp.append(net.batchvar[m]) # torch shape (input_channels,)
                    m = m + 1
    else:
        for param in net.parameters():
            # make conv layers into matrices with np shape (output_channels,input_channels)
            # and batchnorm layers into matrices with np shape (input_channels, 1)
            temp.append(param.view(param.data.shape[0], -1).cpu())
        m = 0
        for module in net.modules():
            if hasattr(module, "running_mean"):
                if evalmode:
                    ttemp.append(module.running_mean.cpu()) # torch shape (input_channels,)
                    ttemp.append(module.running_var.cpu()) # torch shape (input_channels,)
                else:
                    ttemp.append(net.batchmean[m].cpu()) # torch shape (input_channels,)
                    ttemp.append(net.batchvar[m].cpu()) # torch shape (input_channels,)
                    m = m + 1

    # Network-specific allocations of weights into the tensors L^l
    L = []
    for n in range(9):
        w = torch.einsum('ij,i->ij', temp[3 * n], 1.0 / np.sqrt(ttemp[2 * n + 1] + 1e-10)) # torch shape (i,1)
        if n == 0:
            w = w / 2.**0.5

        # Build tensor w
        m1 = torch.diagflat(w[:,0])
        m2 = torch.zeros(w.shape[0], 1)
        m3 = torch.zeros(1, w.shape[0])
        m4 = torch.ones(1,1)
        r1 = torch.cat((m1, m2), dim = 1) # row 1
        r2 = torch.cat((m3, m4), dim = 1) # row 2
        w = torch.cat((r1,r2), dim = 0)
        w = tn.Node(w)

        z = temp[3 * n + 1] - torch.einsum('ij,i->ij', temp[3 * n], ttemp[2 * n] / np.sqrt(ttemp[2 * n + 1] + 1e-10)) # torch shape (i,1)

        # Build tensor z
        m1 = torch.diagflat(torch.ones(z.shape[0]))
        m2 = z.view(z.shape[0], 1)
        m3 = torch.zeros(1, z.shape[0])
        m4 = torch.ones(1,1)
        r1 = torch.cat((m1, m2), dim = 1) # row 1
        r2 = torch.cat((m3, m4), dim = 1) # row 2
        z = torch.cat((r1,r2), dim = 0)
        z = tn.Node(z)

        if n == 8: # classification layers special case (last L tensor)
            a1 = temp[26] # torch shape (c,i)
            m1 = a1
            m2 = torch.zeros(a1.shape[0], 1)
            m3 = torch.zeros(1, a1.shape[1])
            m4 = torch.ones(1,1)
            r1 = torch.cat((m1, m2), dim = 1) # row 1
            r2 = torch.cat((m3, m4), dim = 1) # row 2
            a1 = torch.cat((r1,r2), dim = 0)
            a1 = tn.Node(a1)
            a = a1
        else:
            a = temp[3 * n + 2] # torch shape (b,i)
            m1 = a
            m2 = torch.zeros(a.shape[0], 1)
            m3 = torch.zeros(1, a.shape[1])
            m4 = torch.ones(1,1)
            r1 = torch.cat((m1, m2), dim = 1) # row 1
            r2 = torch.cat((m3, m4), dim = 1) # row 2
            a = torch.cat((r1,r2), dim = 0)
            a = tn.Node(a)

        a[1] ^ z[0]
        z[1] ^ w[0]
        # tensor indices (B,I), L=a_{B,I''}.z_{I'',I'}.w_{I',I}, which is supposed to act on input column vectors v_I from the left
        Ltemp = tn.contractors.greedy((a, z, w), output_edge_order=(a[0], w[1]))
        Ltemp.name = 'L%s' %n
        L.append(Ltemp)

    # Setup representation functions phi^{s_i}(x_i) into tensors, for given input pixels, s_i taking 3 values
    pixel_size =  x.shape[0] # number of sites at given layer originating from pixels, here == number of given pixels
    phitemp = torch.empty(pixel_size, net.modes + 1)

    if legendre:
        for n in range(net.modes):
            if n == 0:
                phitemp[:,n] = (3./2.)**(1./2.) * torch.from_numpy(x)[:]
            if n == 1:
                phitemp[:,n] = (5./2.)**(1./2.) * (1./2.) * (torch.from_numpy(x)[:]**2 - 1.)
            if n == 2:
                phitemp[:,n] = (7./2.)**(1./2.) * (1./2.) * (5. * torch.from_numpy(x)[:]**3 - 3. * torch.from_numpy(x)[:])
            if n == 3:
                phitemp[:,n] = (9./2.)**(1./2.) * (1./8.) * (35. * torch.from_numpy(x)[:]**4 - 30. * torch.from_numpy(x)[:]**2 + 3.)
            if n == 4:
                phitemp[:,n] = (11./2.)**(1./2.) * (1./8.) * (63. * torch.from_numpy(x)[:]**5 - 70. * torch.from_numpy(x)[:]**3 + 15. * torch.from_numpy(x)[:])
            if n == 5:
                phitemp[:,n] = (13./2.)**(1./2.) * (1./16.) * (231. * torch.from_numpy(x)[:]**6 - 315. * torch.from_numpy(x)[:]**4 + 105. * torch.from_numpy(x)[:]**2 - 5.)
            if n == 6:
                phitemp[:,n] = (15./2.)**(1./2.) * (1./16.) * (429. * torch.from_numpy(x)[:]**7 - 693. * torch.from_numpy(x)[:]**5 + 315. * torch.from_numpy(x)[:]**3 - 35. * torch.from_numpy(x)[:])
            if n == 7:
                phitemp[:,n] = (17./2.)**(1./2.) * (1./128.) * (6435. * torch.from_numpy(x)[:]**8 - 12012. * torch.from_numpy(x)[:]**6 + 6930. * torch.from_numpy(x)[:]**4 - 1260. * torch.from_numpy(x)[:]**2 + 35.)
            if n == 8:
                phitemp[:,n] = (19./2.)**(1./2.) * (1./128.) * (12155. * torch.from_numpy(x)[:]**9 - 25740. * torch.from_numpy(x)[:]**7 + 18018. * torch.from_numpy(x)[:]**5 - 4620. * torch.from_numpy(x)[:]**3 + 315. * torch.from_numpy(x)[:])
            if n == 9:
                phitemp[:,n] = (21./2.)**(1./2.) * (1./256.) * (46189. * torch.from_numpy(x)[:]**10 - 109395. * torch.from_numpy(x)[:]**8 + 90090. * torch.from_numpy(x)[:]**6 - 30030. * torch.from_numpy(x)[:]**4 + 3465. * torch.from_numpy(x)[:]**2 - 63.)
            if n == 10:
                phitemp[:,n] = (23./2.)**(1./2.) * (1./256.) * (88179. * torch.from_numpy(x)[:]**10 - 230945. * torch.from_numpy(x)[:]**8 + 218790. * torch.from_numpy(x)[:]**6 - 90090. * torch.from_numpy(x)[:]**4 + 15015. * torch.from_numpy(x)[:]**2 - 693.)
            if n == 11:
                phitemp[:,n] = (25./2.)**(1./2.) * (1./1024.) * (676039. * torch.from_numpy(x)[:]**12 - 1939938. * torch.from_numpy(x)[:]**10 + 2078505. * torch.from_numpy(x)[:]**8 - 1021020. * torch.from_numpy(x)[:]**6 + 225225. * torch.from_numpy(x)[:]**4 - 18018. * torch.from_numpy(x)[:]**2 + 231.)
            if n == 12:
                phitemp[:,n] = (27./2.)**(1./2.) * (1./1024.) * (1300075. * torch.from_numpy(x)[:]**12 - 4056234. * torch.from_numpy(x)[:]**10 + 4849845. * torch.from_numpy(x)[:]**8 - 2771340. * torch.from_numpy(x)[:]**6 + 765765. * torch.from_numpy(x)[:]**4 - 90090. * torch.from_numpy(x)[:]**2 + 3003.)
            if n == 13:
                phitemp[:,n] = (29./2.)**(1./2.) * (1./2048.) * (5014575. * torch.from_numpy(x)[:]**14 - 16900975. * torch.from_numpy(x)[:]**12 + 22309287. * torch.from_numpy(x)[:]**10 - 14549535. * torch.from_numpy(x)[:]**8 + 4849845. * torch.from_numpy(x)[:]**6 - 765765. * torch.from_numpy(x)[:]**4 + 45045. * torch.from_numpy(x)[:]**2 - 429.)
            if n == 14:
                phitemp[:,n] = (31./2.)**(1./2.) * (1./2048.) * (9694845. * torch.from_numpy(x)[:]**14 - 35102025. * torch.from_numpy(x)[:]**12 + 50702925. * torch.from_numpy(x)[:]**10 - 37182145. * torch.from_numpy(x)[:]**8 + 14549535. * torch.from_numpy(x)[:]**6 - 2909907. * torch.from_numpy(x)[:]**4 + 255255. * torch.from_numpy(x)[:]**2 - 6435.)
            if n == 15:
                phitemp[:,n] = (33./2.)**(1./2.) * (1./32768.) * (300540195. * torch.from_numpy(x)[:]**16 - 1163381400. * torch.from_numpy(x)[:]**14 + 1825305300. * torch.from_numpy(x)[:]**12 - 1487285800. * torch.from_numpy(x)[:]**10 + 669278610. * torch.from_numpy(x)[:]**8 - 162954792. * torch.from_numpy(x)[:]**6 + 19399980. * torch.from_numpy(x)[:]**4 - 875160. * torch.from_numpy(x)[:]**2 + 6435.)
            if n == 16:
                phitemp[:,n] = (35./2.)**(1./2.) * (1./32768.) * (583401555. * torch.from_numpy(x)[:]**16 - 2404321560. * torch.from_numpy(x)[:]**14 + 4071832900. * torch.from_numpy(x)[:]**12 - 3650610600. * torch.from_numpy(x)[:]**10 + 1859107250. * torch.from_numpy(x)[:]**8 - 535422888. * torch.from_numpy(x)[:]**6 + 81477396. * torch.from_numpy(x)[:]**4 - 5542680. * torch.from_numpy(x)[:]**2 + 109395.)
            if n == 17:
                phitemp[:,n] = (37./2.)**(1./2.) * (1./65536.) * (2268783825. * torch.from_numpy(x)[:]**18 - 9917826435. * torch.from_numpy(x)[:]**16 + 18032411700. * torch.from_numpy(x)[:]**14 - 17644617900. * torch.from_numpy(x)[:]**12 + 10039179150. * torch.from_numpy(x)[:]**10 - 3346393050. * torch.from_numpy(x)[:]**8 + 624660036. * torch.from_numpy(x)[:]**6 - 58198140. * torch.from_numpy(x)[:]**4 + 2078505. * torch.from_numpy(x)[:]**2 - 12155.)
            if n == 18:
                phitemp[:,n] = (39./2.)**(1./2.) * (1./65536.) * (4418157975. * torch.from_numpy(x)[:]**18 - 20419054425. * torch.from_numpy(x)[:]**16 + 39671305740. * torch.from_numpy(x)[:]**14 - 42075627300. * torch.from_numpy(x)[:]**12 + 26466926850. * torch.from_numpy(x)[:]**10 - 10039179150. * torch.from_numpy(x)[:]**8 + 2230928700. * torch.from_numpy(x)[:]**6 - 267711444. * torch.from_numpy(x)[:]**4 + 14549535. * torch.from_numpy(x)[:]**2 - 230945.)
        phitemp[:, -1] = 1. / (2.**(0.5)) #/ cc1
    else:
        for n in range(int(net.modes/2)):
            phitemp[:, 2*n] =  2.**(1./2.) *  torch.cos(2. * math.pi * (n+1) * torch.from_numpy(x)[:])
            phitemp[:, 2*n+1] =  2.**(1./2.) *  torch.sin(2. * math.pi * (n+1) * torch.from_numpy(x)[:])
        phitemp[:, -1] = 1.
        

    phi = [] # to hold vector node for each pixel below, i.e. (2*freq+1,1)-tensor in order to do contractions below
    for i in range(phitemp.shape[0]):
        phi.append(tn.Node(phitemp[i,:].view(net.modes + 1,1), name='phi%s' %(i+1)))

    # Inputs for first hidden layer, no free indices in the middle, shape (B,B~), where ~ denotes adjoint index
    # g1[i] contains the contraction with the pixel x_i, g2 is the rest (same in all sites due to weight sharing)
    g1 = []
    for i in range(pixel_size):
        Lcopy = list(tn.copy((L[0],))[0].values())[0]
        phicopy = list(tn.copy((phi[i],))[0].values())[0]

        L[0][1] ^ phi[i][0]
        phi[i][1] ^ phicopy[1]
        phicopy[0] ^ Lcopy[1]

        g1.append(tn.contractors.greedy((L[0], phi[i], phicopy, Lcopy), output_edge_order=(L[0][0], Lcopy[0])))
    if pixel_size < 256:
        Lcopy = list(tn.copy((L[0],))[0].values())[0]

        L[0][1] ^ Lcopy[1]

        g2 = tn.contractors.greedy((L[0], Lcopy), output_edge_order=(L[0][0], Lcopy[0]))

    for l in range(max_it):
        gg1 = [] # to contain contractions of this layer originating from given pixels
        if pixel_size % 2 == 0: # set flag for odd number of pixel-sites at previous level
            odd = False
        else:
            odd = True
        pixel_size = int(math.ceil(pixel_size / 2)) # new number of sites originating from pixels at this layer, we need ceiling since we contract every 2 pixels
        
        # Loop over sites originating from pixels
        for i in range(pixel_size):
            if i == pixel_size - 1 and odd:
                gg1.append(contr(L[l + 1], g1[2*i], g2))  # full inner contractions with order-3 delta tensor
            else:
                gg1.append(contr(L[l + 1], g1[2*i], g1[2*i+1])) # full inner contractions with order-3 delta tensor
        # Rest of contractions (with no pixels), only if needed
        if pixel_size < 2**(8-(l+1)):
            g2copy = list(tn.copy((g2,))[0].values())[0]
            gg2 = contr(L[l + 1], g2, g2copy)

        g1 = gg1
        if pixel_size < 2**(8-(l+1)):
            g2 = gg2

    if len(g1) == 1:
        return g1[0]
    elif half:
        return g2, L[l+2]
    else:
        return g2


def EEhalf(net, c0 = 1./math.sqrt(2), svd=False, prt=False):
    """Computes entanglement entropy of bipartition."""
    # calculate N = Tr_B(|phi_B^b> <phi_B^b'|) = <phi_B^b|phi_B^b'> total contraction with free outer legs only
    #N = marginal_contr(net, half = True, prt = prt).tensor.detach().numpy()
    with torch.no_grad():
        N, L = marginal_contr(net, half = True) # N shape (B, ~B, free indices), L shape (Y,B), with Y->(y,1)
        N = N.tensor.detach().numpy()
        L = L.tensor.detach().numpy()

    # diagonalize N = P.D.Pinv, Pdagg = Pinv unitary
    d, P = np.linalg.eigh(N)
    Pinv = np.linalg.inv(P)
    Pdagg = np.transpose(np.conjugate(P))
    errorP = np.amax(np.abs(Pdagg - Pinv))
    if prt:
        print("Unitarity error in P:", errorP)
        print("min Pdagg:", np.amin(np.abs(Pdagg)),"max Pdagg", np.amax(np.abs(Pdagg)))
        print("min Pinv:", np.amin(np.abs(Pinv)),"max Pinv", np.amax(np.abs(Pinv)))
        print()

    # define U := P.Dsq.Pinv so that N = U.Udagg, same shape as N
    Dsq = np.diag(np.sqrt(np.abs(d)))
    U = contract(P, [0,1], Dsq, [1,2], Pdagg, [2,3], [0,3])
   
    if prt:
        Udagg = np.transpose(np.conjugate(U))
        errorU = np.amax(np.abs(N - contract(U, [0,1], Udagg, [1,2], [0,2])))
        print("Nmin:", np.amin(np.abs(N)), ", error in N=U.U^dagger:", errorU, "Nmax:", np.amax(np.abs(N)))
        print()

    # define Ntilde where (Ntilde)_{B,B~}^Y = N_{B,B~} * a_{B}^Y * a_{B~}^Y, for each Y (no summation)
    Ntilde = np.empty((L.shape[0], L.shape[1], L.shape[1])) # shape (Y,B,B~), Y runs over labels + 1 elements
    for Y in range(L.shape[0]):
        Ntilde[Y, :, :] = N
        for B in range(L.shape[1]):
            for Badj in range(L.shape[1]):
                Ntilde[Y, B, Badj] = Ntilde[Y, B, Badj] * L[Y, B] * L[Y, Badj]

    # define M_{E,E~}^Y = Ntilde_{B,B~}^Y * U_{B,E} * Udagg_{B~,E~} (B,B~ contractions), M = U^T.Ntilde.U*
    M = contract(np.transpose(U), [0,1], Ntilde, [2,1,3], np.conjugate(U), [3,4], [2,0,4])

    for y in range(M.shape[0] - 0):
        M[y] = M[y] / np.trace(M[y])

    # diagonalize M^Y = V^Y.S^Y.Vinv^Y
    if svd:
        u, s, vh = np.linalg.svd(M, compute_uv = True)
    else:
        s, V = np.linalg.eigh(M)
    
    if prt:
        Vinv = np.linalg.inv(V[0])
        Vdagg = np.transpose(np.conjugate(V[0]))
        errorV = np.amax(np.abs(Vdagg - Vinv))
        print("Unitarity error in V:", errorV)
        print("min Vdagg:", np.amin(np.abs(Vdagg)),"max Vdagg", np.amax(np.abs(Vdagg)))
        print("min Vinv:", np.amin(np.abs(Vinv)),"max Vinv", np.amax(np.abs(Vinv)))
        print()

        id = contract(V[0],[0,1], Vdagg,[1,2], [0,2])
        print("V.Vdagg min/max:", np.amin(id), np.amax(id))


    # chop extremely small eigenvalues and verify that remaining ones are positive
    if not svd:
        for y in range(s.shape[0]):
            for i in range(s.shape[1]):
                if np.abs(np.imag(s[y, i])) != 0 and np.abs(np.imag(s[y, i])) < 10e-8:
                    if np.real(s[y, i]) >= 0:
                        s[y, i] = np.real(s[y, i])
                    elif np.abs(np.real(s[y, i])) < 10e-8:
                        s[y, i] = 0.0
                        ff=1
                    else:
                        raise Exception("negative eigenvalue...", s[y, i])
                elif np.real(s[y, i]) < 0:
                    if np.abs(np.real(s[y, i])) < 10e-8:
                        s[y, i] = 0.0
                        ff=1
                    else:
                        raise Exception("negative eigenvalue...", s[y, i])
        s = np.real(s)

    # calculate EE based on eigenvalues s_i, and normalize
    p = np.empty_like(s)
    ee = []
    for y in range(s.shape[0]):
        res = 0
        norm = 0
        for i in range(s.shape[1]):
            norm = norm + s[y, i]
        for i in range(s.shape[1]):
            if s[y, i] == 0:
                p[y, i] = 0
                continue
            else:
                res = res - (s[y, i] / norm) * math.log(s[y, i] / norm)
                p[y, i] = s[y, i] / norm
        if prt:
            print("EE", y, ": ", round(res, 10))
        ee.append(res)

    if prt:
        print("max EE:", np.round(math.log(len(V[0])), 5))

    return ee


