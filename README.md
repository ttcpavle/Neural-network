# Neural network i backpropagation

Za razumevanje algoritma backpropagation potrebno je poznavanje funkcionisanja jednog neurona i strukture mreže neurona. Takođe potrebno je poznavanje koncepata poput izvoda, chain-rule pravila i vektora gradijenta.

Oznake korišćene u ovom objašnjenju:
- $x$ - ulazni vektor
- $y$ - target vektor
- $C$ - loss funkcija
- $l$ - posmatrani sloj neurona
- $L$ - broj slojeva
- $W^L$ - matrica težina između slojeva l-1 i l.
- $w^l_{jk}$ - težina veze između neurona j u sloju $l$ i neurona k u sloju l-1
- $f^l$ - aktivaciona funkcija sloja l
- $a^l_j$ - aktivacija neurona j u sloju l
- $\delta^l$ - delta vektor sloja l
- $\odot$ - hadamardov proizvod (element-wise množenje)

Neuronska mreža je zapravo funkcija oblika:

$$g(x)=f^L(W^Lf^{L-1}(W^{L-1}...f^1(W^1x)...))$$

Skup za treniranje je skup input-output parova:

$$\{(x_i,y_i)\}$$

Loss (gubitak) modela za jedan par $(x_i,y_i)$ je:

$$C(y_i, g(x_i))$$

## Backpropagation

Backpropagation je metoda koja se koristi za račun gradijenta cost (loss) funkcije za svaki par ulaz-izlaz $(x_i,y_i)$. Gradijent cost funkcije se računa prema svakoj težini $w^l_{jk}$ i bias-u $b^l_j$: $$\frac{\partial C}{\partial \omega^l_{jk}}$$ Ovi parcijalni izvodi mogu da se računaju pomoću chain-rule pravila, ali ovo je vrlo neefikasno. Zato se koristi dinamičko programiranje i računa gradijent svakog sloja (po ulazu). Ovaj gradijent označavamo sa $\delta^l$.
Ako bi računali delte od prvog sloja, imali bi redundantni račun jer delta trenutnog sloja $l$ sadrži delte narednih slojeva $l+1, l+2, ..., L$. gde je L poslednji sloj.

Počećemo od totalnog diferencijala i doći do efikasnog algoritma za račun ovog parcijalnog izvoda.

## Račun delti

Neuronska mreža je kompozicija funkcija. Da se podsetimo nekih pravila za kompoziciju:
Ako imamo kompoziciju 2 diferencijabilne funkcije $h(x) = z(y(x))$ tj $h=z \circ y$  tada za svako x važi:
$h^\prime (x) = z^\prime(y(x))y^\prime(x)$ - prema langranžovoj notaciji

Nadalje će se koristiti Leibniz notacija:

$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$

___

Koristeći pravilo 

$$D(f \circ g)_a = Df_{g(a)} \cdot Dg_a$$

za neku tačku a, možemo reći da je totalni diferencijal (total derivative) cost funkcije jednak:

$$\frac{dC}{dx} = \frac{dC}{da^L} \cdot \frac{da^L}{dz^L} \cdot \frac{dz^L}{da^{L-1}} \cdot \frac{da^{L-1}}{dz^{L-1}} \cdot \frac{dz^{L-1}}{da^{L-2}} \dots \frac{da^1}{dz^1} \cdot \frac{dz^1}{dx}$$

Totalni diferencijal neke funkcije predstavlja najbolju linearnu aproksimaciju te funkcije u okolini neke tačke.
Možemo primetiti sledeće:
- Svaki član $\frac{da^l}{dz^l}$ je isto što i $f'(z^l)$ tj u kraćoj oznaci $(f^l)'$
- Svaki član $\frac{dz^l}{da^{l-1}}$ je isto što i $W^l$ jer $z^l = W^l a^{l-1} + b^l$

Kada ovo zamenimo:

$$\frac{dC}{dx} = \frac{dC}{da^L} \odot (f^L)' \cdot W^L \odot (f^{L-1})' \cdot W^{L-1} \odot \dots \odot (f^1)' \cdot W^1$$

Pri čemu je $\odot$ hadamardov proizvod (elementwise množenje). Zašto ta operacija? Za svaku aktivaciju u sloju l važi: $$a_i^l = f(z_i^l)$$ To znači da $a_1$ zavisi **samo** od $z_1$, $a_2$ samo od $z_2$, itd. Ako bismo napisali punu Jakobijanovu matricu $\frac{da^l}{dz^l}$ (tj prikazali zavisnost aktivacije svakog neurona po ulaznim sumama za sve neurone), ona bi bila **dijagonalna**

$$\frac{da^l}{dz^l} = \left[\begin{matrix} f'(z_1^l) & 0 & \dots \cr 0 & f'(z_2^l) & \dots \cr \vdots & \vdots & \ddots \end{matrix}\right]$$

Množenje vektora sa dijagonalnom matricom poput ove (matrično množenje) može se zapisati drugačije tj, ako imamo $D \cdot v$ gde je v vektor a D dijagonalna matrica, to je isto kao i $d \odot v$ gde je $d$ 1D vektor sa elementima dijagonalne matrice $D$

Cost funkcija je $C: R^n \rightarrow R$ i njen totalni diferencijal je jakobijeva matrica sa jednim redom, koja kada se transponuje daje vektor gradijenta.
Primenom pravila: $(A \cdot B)^T = B^T \cdot A^T$ dobija se gradijent cost funkcije prema ulazu x: 

$$\nabla_x C = (W^1)^T \cdot (f^1)' \odot \dots \odot (W^{L-1})^T \cdot (f^{L-1})' \odot (W^L)^T \cdot (f^L)' \odot \nabla_{a^L} C$$

Sada možemo definisati delta za poslednji sloj L koji sadrži $\nabla_{a^L} C$:

$$\delta^L = (f^L)' \odot \nabla_{a^L} C$$

Odatle sledi da je delta za bilo koji sloj $l$ jednaka:

$$\delta^l = ((W^{l+1})^T \delta^{l+1}) \odot f'(z^l)$$

Čime smo pokazali vezu između delti susednih slojeva i uspostavili rekurentnu relaciju.

## Izvod po parametrima modela

Ove delte koristiće nam kod nalaženja izvoda po težinama. Možemo napomenuti da u forward pass-u parametri se ponašaju kao konstante a ulaz je promenljiva koju menjamo, a u backward passu je obrnuto.

Dakle, želimo da formiramo gradijent funkcije $C$ kako bi smo mogli da je minimiziramo (koristeći gradient descent). Da bi to uradili trebaju nam uticaji trenutnih parametara modela (weights and biases) na cost, tj parcijalni izvodi po tim parametrima. Ovako bi to izgledalo kada ne bi imali skrivene slojeve:

$$\frac{\partial C}{\partial w} = \frac{\partial C}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$


Prva dva množioca predstavljaju deltu $\delta^l = \frac{\partial C}{\partial z^l}$ koja predstavlja jedinični uticaj ulazne sume na krajnju grešku, a poslednji član predstavlja aktivacije prethodnog sloja, jer izvod $w_1a_1+w_2a_2+...+w_ka_k $ po $w_i$ je samo $a_i$. Ako bi imali dublju neuronsku mrežu, tada bi ovaj izvod izgledao ovako:

$$\frac{\partial C}{\partial w_{jk}^l} = \underbrace{\frac{\partial C}{\partial a^L} \cdot \frac{\partial a^L}{\partial z^L} \dots \frac{\partial a^l}{\partial z^l}}_{\delta_j^l} \cdot \underbrace{\frac{\partial z^l}{\partial w_{jk}^l}}_{a_k^{l-1}}$$

**Dakle da bi našli izvod cost funkcije po nekoj težini, treba nam delta neurona narednog sloja i aktivacija neurona prethodnog sloja.**

- Dakle, uzev u obzir funkciju cilja i aktivacionu funkciju ovog sloja neurona (tj njihove izvode), vrednosti $\delta$ pokazuju koliko ulazna suma za sloj neurona utiče na grešku.


**Ovde nastaje rekurzivna backpropagation logika. Pošto aktivacije nekog sloja $l-1$ utiču na aktivacije trenutnog sloja $l$ pa samim tim i na vrednost cost funkcije, a aktivacije sloja $l-1$ direktno zavise od ulazne sume u sloj $l-1$ koja zavisi od aktivacija sloja $l-2$, možemo primetiti uticaj aktivacija svih slojeva neurona u mreži na cost funkciju. Ovakav tip problema rešava se dinamičkim programiranjem koji koristi backpropagation**


## Ažuriranje parametara

Znajući osetljivost neurona na ulaznu sumu (delta), i znajući težine i aktivacije koje učestvuju u toj ulaznoj sumi, možemo promeniti te težine tako da se **poprave srazmerno deltama i aktivacijama**. Što su veće delte i aktivacije, to će više da smanjimo vrednost odgovarajuće težine i time smanjimo vezu između neurona j i k.

$$w_{jk}^l \leftarrow w_{jk}^l - \eta \cdot \frac{\partial C}{\partial w_{jk}^l}$$

gde je $\eta$ hiperparametar za skaliranje koraka, i važi:

$$\frac{\partial C}{\partial w_{jk}^l} = \delta_j^l \cdot a_k^{l-1}$$

Što smo ranije pokazali.

---

Sada imamo 4 ključne jednačine za backpropagation:

$$\delta^L = (f^L)' \odot \nabla_{a^L} C$$

$$\delta^l = ((W^{l+1})^T \delta^{l+1}) \odot f'(z^l)$$

$$w_{jk}^l \leftarrow w_{jk}^l - \eta \cdot \delta_j^l \cdot a_k^{l-1}$$

$$\delta^l = \frac{\partial C}{\partial z^l}$$

## U ovom kodu

U ovom kodu parametri modela se ažuriraju odmah za svaki datapoint:
1) Napravimo neuronsku mrežu i definišemo njene aktivacione funkcije.
2) Uradimo jedan forward pass za određeni datapoint (može i batch) i sačuvamo aktivacije neurona i ulazne sume (težinske sume).
3) U zavisnosti od izabrane cost funkcije, izračunamo gradijent cost funkcije koristeći dobijene aktivacije u zadnjem sloju i target. Potom koristimo f'(z^L) i izračunamo zadnji delta
4) za svaki sledeći (skriveni) sloj računamo $\delta^l$ 
5) za taj sloj l ažuriramo težine po formuli

## Dodatni resursi

[Neuralne mreze link 1](http://neuralnetworksanddeeplearning.com/chap2.html)

[3Blue1Brown backpropagation](https://youtu.be/Ilg3gGewQ5U?si=T3WD7m91dNZafIkb)

[wikipedia](https://en.wikipedia.org/wiki/Backpropagation)
