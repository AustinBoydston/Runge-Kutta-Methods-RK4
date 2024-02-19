#Austin Boydston
#Numerical Analysis 
#Homework week 11

import numpy as np
import matplotlib.pyplot as plt


#A function
def fun_a(x, t):
    return 1 + (t-x)*(t-x)

#Partial derivative in terms of x
def fun_ax(x, t):
    return -2 * (t-x)
#Partial derivative in terms of t
def fun_at(x, t):
    return 2 * (t - x) 
#Second derivatives
def fun_att(x, t):
    return 2
def fun_axx(x, t):
    return 2;
def fun_atx(x, t):
    return -2;

def taylor_order2(fun, tfun, ttfun,xfun, xxfun, txfun, a, b, h, n, alpha):
    omega = alpha
    for i in range (n):
        ti = a + i*h
        omega = omega + tfun(alpha, a) * (ti-a) + xfun(alpha, a) * (omega-alpha) + xxfun(alpha, a)*(omega-alpha)*(omega-alpha)*0.5 + ttfun(alpha, a) * (ti-a)*(ti-a) * .5 + txfun(alpha, a) * (ti-a)*(omega-alpha) 
        print(omega)
    return omega
   

def taylor_order1(fun, tfun, xfun, a, b, h, n, alpha):
    omega = alpha
    for i in range (n):
        ti = a + i*h
        omega = omega + tfun(alpha, a) * (ti-a) + xfun(alpha, a) * (omega-alpha) 
        print(omega)
    return omega

#Part B
def fun_b(x, t):
    return np.sin(2*t)/(t*t) - (2*x)/t

# Derivative in terms of t
def fun_bx(x, t):
    return(-2/t)
def fun_bt(x, t):
   # return (2*np.cos(2*t) *t*t - 2*t*np.sin(2*t))/(t*t*t*t) + (2*x)/(t*t)
    return -2*(np.sin(2*t) - t*np.cos(2*t) - t)/(t*t*t)
#second derivatives
def fun_bxx(x, t):
    return 0
def fun_bxt(x, t):
    return 2/(t*t)
def fun_btt(x, t):
    return -((4*t*t - 6)*np.sin(2*t) + 8 * t * np.cos(2*t) + 4*t*x)/(t*t*t*t)
#Third Derivatives
def fun_bxxx(x, t):
    return 0
def fun_bxxt(x, t):
    return 0
def fun_bxtx(x, t):
    return 0
def fun_btxt(x, t):
    return -4* (1/(t*t*t))
def fun_bxtt(x, t):
    return -4/(t*t*t*t)
def fun_btxx():
    return 0
def fun_bttx(x, t):
    return (4/(t*t*t))
def fun_bttt(x, t):
    return -((24*t*t-24)*np.sin(2*t)+(36*t-8*t*t*t)*np.cos(2*t)+12*t*x)/(t*t*t*t*t)
#Forth Derivatives
def fun_btxtt(x, t):
    return 12/(t*t*t*t)
def fun_bxttt(x, t):
    return 12 / (t*t*t*t)
def fun_bttxt(x, t):
    return (-12/t*t*t*t)
def fun_btttx(x, t):
    return 12/(t*t*t*t)
def fun_btttt(x, t):
    return (-(16*t*t*t*t - 144*t*t + 120)* np.sin(2*t) + (64*t*t*t-192*t)*np.cos(2*t)-48*t*x)/(t*t*t*t*t*t)

def taylor_order4(fun, tfun, ttfun,xfun, xxfun, txfun, a, b, h, n, alpha, xttfun, txtfun, ttxfun, tttfun, txttfun, xtttfun, ttxtfun, ttttfun):
    omega = alpha
    for i in range (n):
        ti = a + i*h
        omega = omega + tfun(alpha, a) * (ti-a) + xfun(alpha, a) * (omega-alpha) + xxfun(alpha, a)*(omega-alpha)*(omega-alpha)*0.5 + ttfun(alpha, a) * (ti-a)*(ti-a) * .5 + txfun(alpha, a) * (ti-a)*(omega-alpha) 
        omega = omega + xttfun(alpha, a) * (omega - alpha) * (ti-a) * (ti-a) + txtfun(alpha, a)* (ti-a)*(omega - alpha)*(ti-a) + ttxfun(alpha, a) *(ti-a)*(ti-a)*(omega-alpha) + tttfun(alpha, a) *(ti-a)*(ti-a)*(ti-a) + txttfun(alpha, a) *(ti-a)*(omega-alpha)*(ti-a) *(ti-a) + xtttfun(alpha, a) *(omega - alpha) *(ti-a)*(ti-a)*(ti-a) + ttxtfun(alpha, a) *(ti-a)*(ti-a)*(omega - alpha) *(ti-a) + ttttfun(alpha, a) *(ti-a)*(ti-a)*(ti-a)*(ti-a)
        print(omega)
    return omega

#######Kuttas#########
#Runge kutta k1 part A
def kutta_1a(t, x, h):
    return h * (1+(t-x)*(t-x))
#Runge kutta k2 part A
def kutta_2a(t, x, h, k1):
    return h * (1+((t+h) - (x+k1(t, x, h)) )  * ((t+h) - (x+k1(t, x, h)) ))

#Runge Kutta O4 k2 part A
def kutta_O4_2a(t, x, h, k1):
    return h * (1+((t+h*.5) - (x+.5*k1(t, x, h))) * ((t+h*.5) - (x+.5*k1(t, x, h))) )

#Runge kutta O4 k3 part A
def kutta_O4_3a(t, x, h, k2):
    return h * (1+((t+h*.5) - (x+.5*k2(t, x, h, kutta_1a))) * ((t+h*.5) - (x+.5*k2(t, x, h, kutta_1a))) )

#Runge kutta O4 k4 part A
def kutta_O4_4a(t, x, h, k3):
    return h * (1+((t+h) - (x+k3(t, x, h, kutta_O4_2a))) * ((t+h) - (x+k3(t, x, h, kutta_O4_2a))) )



#Runge Kutta k1 part B
def kutta_1b(t, x, h):
    return h * (1/(t*t)) * (np.sin(2*t)-2*t*x)
#Runge Kutta k2 part B
def kutta_2b(t, x, h, k1):
    return h * (1/((t+h)*(t+h))) * (np.sin(2*(t+h))-2*(t+h)*(x+k1(t, x, h)))

#Runge Kutta O4 k2 part B
def kutta_O4_2b(t, x, h, k1):
        return h * (1/((t+.5*h)*(t+.5*h))) * (np.sin(2*(t+h*.5))-2*(t+h*.5)*(x+.5*k1(t, x, h)))

#Runge kutta O4 k3 part B
def kutta_O4_3b(t, x, h, k2):
        return h * (1/((t+h*.5)*(t+h*.5))) * (np.sin(2*(t+h*.5))-2*(t+h*.5)*(x+.5*k2(t, x, h, kutta_1b)))

#Runge kutta O4 k4 part B
def kutta_O4_4b(t, x, h, k3):
        return h * (1/((t+h)*(t+h))) * (np.sin(2*(t+h))-2*(t+h)*(x+k3(t, x, h, kutta_O4_2b)))


#Runge Kutta order 2
def Runge_Kutta2(k1, k2,a, alpha, h, n):
    omega = alpha
    for i in range(n):
        ti = a + i*h
        omega = omega + 0.5*(k1(ti, omega, h)+ k2(ti, omega, h, k1))
        print(omega)
    return omega

#Runge Kutta order 4
def Runge_Kutta4(k1, k2, k3, k4, a, alpha, h, n):
    omega = alpha
    for i in range (n):
        ti = a + i * h
        omega = omega + (1/6)* (k1(ti, omega, h) + 2*k2(ti, omega, h, k1) + 2*k3(ti, omega, h, k2)+ k4(ti, omega, h, k3))
        print(omega)

    return omega

#####################C1#######################
print('#################C1###################')
print('testing taylor series order 2 on part A')
taylor_order2(fun_a, fun_at, fun_att, fun_ax, fun_axx,fun_atx, 2, 3, 0.05, 20, 1)
print()
print('Testing taylor series order 4 on part A')
print('Same as order 2 since second derivatives are constants')
taylor_order2(fun_a, fun_at, fun_att, fun_ax, fun_axx,fun_atx, 2, 3, 0.05, 20, 1)
print()

print('Testing taylor series order 2 on part B')
#print(fun_btt(1, 2))
taylor_order2(fun_b, fun_bt, fun_btt, fun_bx, fun_bxx,fun_bxt, 1, 2, 0.025, 40, 2)
print()
print('Testing taylor series order 4 on part B')
print(taylor_order4(fun_b, fun_bt, fun_btt, fun_bx, fun_bxx, fun_bxt, 1, 2, 0.025, 40, 2, fun_bxtt, fun_btxt, fun_bttx, fun_bttt, fun_btxtt, fun_bxttt, fun_bttxt, fun_btttt))

#Runge Kutta tests
print()
print('Testing Runge Kutta order 2 on part A')
Runge_Kutta2(kutta_1a, kutta_2a, 2, 1, .05, 20)
print()
print('Testing Runge Kutta order 4 on part A')
Runge_Kutta4(kutta_1a, kutta_O4_2a, kutta_O4_3a, kutta_O4_4a, 2, 1, .05, 20)
#
print()
print('Testing Runge Kutta order 2 on part B')
Runge_Kutta2(kutta_1b, kutta_2b, 1, 2, 0.025, 40)
print()
print('Testing Runge Kutta order 4 on part B')
Runge_Kutta4(kutta_1b, kutta_O4_2b, kutta_O4_3b, kutta_O4_4b, 1, 2, 0.025, 40)
print()

print('#############C2##############')
print('=====Part A=====')
print('Runge-Kutta output for part A.')
Runge_Kutta2(kutta_1a, kutta_2a, 2, 1, .01, 100)
print()
print('Runge Kutta order 4 on part A')
Runge_Kutta4(kutta_1a, kutta_O4_2a, kutta_O4_3a, kutta_O4_4a, 2, 1, .01, 100)
print()
print('Euler method for Part A')
print(taylor_order1(fun_a, fun_at, fun_ax, 2, 3, .01, 100, 1))
print()
print()

print('====Part B====')
print('Runge Kutta order 2 on part B')
Runge_Kutta2(kutta_1b, kutta_2b, 1, 2, 0.01, 100)
print()
print('Runge Kutta order 4 on part B')
Runge_Kutta4(kutta_1b, kutta_O4_2b, kutta_O4_3b, kutta_O4_4b, 1, 2, 0.01, 100)
print()
print('Euer method on part B')
print(taylor_order1(fun_b, fun_bt, fun_bx, 1, 2, .01, 100, 2))
print()

print('The Runge-Kutta methods are more precise than Eulers because they converge to the root faster.')
print()









###########C3####################
print('###########C3#############')
#Function f
def fun_c3(t, x):
    return 10*x + 11*t-5*t*t-1

#Solution function
def fun_solc3(t, alpha):
    return alpha * np.power(np.e, 10 * t) + t*t/2 - t

#Kuttas for C3
def kutta1_c3(t, x, h):
    return h * fun_c3(t, x)

def kutta2_c3(t, x, h, k1):
    return h*fun_c3(t+.5*h, x + .5*k1(t, x, h))

def kutta3_c3(t, x, h, k2):
    return h*fun_c3(t+.5*h, x + .5* k2(t, x, h, kutta1_c3))

def kutta4_c3(t, x, h, k3):
    return h * fun_c3(t+h, x + k3(t, x, h, kutta2_c3))

def Runge_Kutta4_plot(k1, k2, k3, k4, a, alpha, h, n, sol_fun):
    omega = alpha
    x = np.array([])
    y = np.array([])
    sol = np.array([])
    for i in range (n):
        ti = a + i * h
        omega = omega + (1/6)* (k1(ti, omega, h) + 2*k2(ti, omega, h, k1) + 2*k3(ti, omega, h, k2)+ k4(ti, omega, h, k3))
        if(i % 10 == 0):
            x = np.append(x, ti)
            y = np.append(y, omega)       
            sol = np.append(sol, sol_fun(ti, alpha))
      #  print(omega, ', ', ti)
    #print(omega)
    return omega, x, y, sol

print('===Numerical Solution (Runge-Kutta)===')
n_sol, x, y, a_sol = Runge_Kutta4_plot(kutta1_c3, kutta2_c3, kutta3_c3, kutta4_c3, 0, 0,(1/256), 768, fun_solc3)
print(n_sol)
print('=====Actual Solution======')
print(fun_solc3(3, 0))
print()
print('They are vastly different. this is because the solution curve for our initial conditions requires very precise input values. If the alpha diviates by any amount what so ever, our solution curve changes because it includes the term "Epsilon * e^10t" where epsilon = alpha. Of course, because we are using a computer, there is inevitably error included in our Runge-Kutta approximation, and therefore our approximation jumps away from the actual solution.')
print('We might describe this behavior as unstable.')
print()
#print(x)
#print(y)

plt.figure(figsize=(10, 10))
plt.title('C3 Alpha = 0')
plt.plot(x, y)
plt.plot(x, a_sol)


print('in order to verify this we simply change alpha to another value. Notice in figure 2 we use alpha = 0.0001, and the solutions follow a similar path.')

n_sol, x, y, a_sol = Runge_Kutta4_plot(kutta1_c3, kutta2_c3, kutta3_c3, kutta4_c3, 0, .0001,(1/256), 768, fun_solc3)

plt.figure(figsize=(10, 10))
plt.title('C3 Alpha = 0.0001')
plt.plot(x, y)
plt.plot(x, a_sol)




#C4
def fun_c4(x, t):
    return 100 * (np.sin(t) - x)


#Actual Solution for comparison purposes
def fun_solc4(x, t):
    return 10000*np.sin(t)/10001 - 100*np.cos(t)/10001 + 100/(10001*np.power(np.e, 100*t));

#Kuttas
def kutta1_c4(t, x, h):
    return h * fun_c4(t, x)

def kutta2_c4(t, x, h, k1):
    return h*fun_c4(t+.5*h, x + .5*k1(t, x, h))

def kutta3_c4(t, x, h, k2):
    return h*fun_c4(t+.5*h, x + .5* k2(t, x, h, kutta1_c4))

def kutta4_c4(t, x, h, k3):
    return h * fun_c4(t+h, x + k3(t, x, h, kutta2_c4))
print()
print('##############C4##################')
print('Yes, The solution should be an occilating curve bounded above and below but the runge-kutta approximation goes far into the negatives. It is different from c3 in that the curves given by the graph curve downward.')

n_sol, x, y, a_sol = Runge_Kutta4_plot(kutta1_c4, kutta2_c4, kutta3_c4, kutta4_c4, 0, 0,0.015, 200, fun_solc4)

plt.figure(figsize=(10, 10))
plt.title('C4 h = 0.015')
plt.plot(x, y)
plt.plot(x, a_sol)

n_sol, x, y, a_sol = Runge_Kutta4_plot(kutta1_c4, kutta2_c4, kutta3_c4, kutta4_c4, 0, 0,0.02, 150, fun_solc4)

plt.figure(figsize=(10, 10))
plt.title('C4 h = 0.02')
plt.plot(x, y)
plt.plot(x, a_sol)

n_sol, x, y, a_sol = Runge_Kutta4_plot(kutta1_c4, kutta2_c4, kutta3_c4, kutta4_c4, 0, 0,0.025, 120, fun_solc4)

plt.figure(figsize=(10, 10))
plt.title('C4 h = 0.025')
plt.plot(x, y)
plt.plot(x, a_sol)

n_sol, x, y, a_sol = Runge_Kutta4_plot(kutta1_c4, kutta2_c4, kutta3_c4, kutta4_c4, 0, 0,0.03, 100, fun_solc4)

plt.figure(figsize=(10, 10))
plt.title('C4 h = 0.03')
plt.plot(x, y)
plt.plot(x, a_sol)


#Show plots
plt.show()