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
- $\odot$ - hadmardov proizvod (element-wise mnozenje)

Neuronska mreza je zapravo funkcija funkcija oblika:
$$g(x)=f^L(W^Lf^{L-1}(W^{L-1}...f^1(W^1x)...))$$
Skup za treniranje je skup input-output parova:
$$\{(x_i,y_i)\}$$
Loss (gubitak) modela za jedan par $(x_i,y_i)$ je:
$$C(y_i, g(x_i))$$

## Backpropagation

Backpropatagion je metoda koja se koristi za račun gradijenta cost (loss) funkcije za svaki par ulaz-izlaz $(x_i,y_i)$. Gradijent cost funkcije se računa prema svakoj tezini $w^l_{jk}$ i bias-u $b^l_j$:$$\partial C / \partial \omega^l_{jk}$$Ovi parcijalni izvodi mogu da se racunaju pomocu chain-rule, ali ovo je vrlo neefikasno. Zato se koristi dinamicko programiranje i racuna gradient svakog sloja (po ulazu). Ovaj gradient oznacavamo sa $\delta^l$.

Ako bi racunali delte od prvog sloja, imali bi redundantni racun jer delta trenutnog sloja $l$ sadrzi delte narednih slojeva $l+1, l+2, ..., L$. gde je L poslednji sloj.
## Račun delti

Neuronska mreza je kompozicija funkcija. Da se podsetimo nekih pravila za kompoziciju:
Ako imamo kompoziciju 2 diferencijabilne funkcije $h(x) = z(y(x))$ tj $h=z \circ y$  tada za svako x vazi:
$h^\prime (x) = z^\prime(y(x))y^\prime(x)$ - prema langranzovoj notaciji

Nadalje ce se koristiti Leibniz notacija:
$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$
___

Dakle, želimo da formiramo gradijent funkcije $C$ kako bi smo mogli da je minimiziramo (koristeći gradient descent). Da bi to uradili trebaju nam uticaji trenutnih parametara modela (weights and biases) na cost, tj parcijalni izvodi po tim parametrima. Želimo da ispitamo osetljivost modela na promene težina.
$$\frac{\partial C}{\partial w} = \frac{\partial C}{\partial a} \cdot \frac{\partial a}{\partial z} \cdot \frac{\partial z}{\partial w}$$
Prva dva mnozioca predstavljaju deltu $\delta^l = \frac{\partial C}{\partial z^l}$ koja predstavlja jedinični uticaj ulazne sume na krajnju grešku, a poslednji član predstavlja aktivacije prethodnog sloja, jer izvod $$w_1a_1+w_2a_2+...+w_ka_k $$ po $w_i$ je $a_i$.

- Dakle, uzev u obzir funkciju cilja i aktivacionu funkciju ovog sloja neurona (tj njihove izvode), vrednosti $\delta$ pokazuju koliko ulazna suma za sloj neurona utice na gresku.
- Pored tezina $w_{kj}$ izmedju slojeva $l-1$ i $l$, aktivacije prethodnog sloja takodje uticu na ulaznu sumu narednog sloja pa samim tim i na cost funkciju. 

**Ovde nastaje rekurzivna backpropagation logika. Posto aktivacije nekog sloja $l-1$ uticu na aktivacije trenutnog sloja $l$ pa samim tim i na vrednost cost funkcije, a aktivacije sloja $l-1$ direktno zavise od ulazne sume u sloj $l-1$ koja zavisi od aktivacija sloja $l-2$, mozemo primetiti uticaj aktivacija svih slojeva neurona u mrezi na cost funkciju. Ovakav tip problema resava se dinamickim programiranjem koji koristi backpropagation**

Poenta je da se u svakoj aktivaciji krije uticaj tezina svih ranijih slojeva. Sada cemo prikazati kompletnu dinamiku mreze.
___
Koristeci pravilo $D(f \circ g )_a = Df_{g(a)} \cdot Dg_a$ za neku tacku a, mozemo reci da je total derivative cost funkcije jednak:
$$\frac{dC}{dx} = \frac{dC}{da^L} \cdot \frac{da^L}{dz^L} \cdot \frac{dz^L}{da^{L-1}} \cdot \frac{da^{L-1}}{dz^{L-1}} \cdot \frac{dz^{L-1}}{da^{L-2}} \dots \frac{da^1}{dz^1} \cdot \frac{dz^1}{dx}$$
Total derivative neke funkcije predstavlja najbolju linearnu aproksimaciju te funkcije u okolini neke tacke.
Možemo primetiti sledeće:
- Svaki član $\frac{da^l}{dz^l}$ je isto što i $f'(z^l)$
- Svaki član $\frac{dz^l}{da^{l-1}}$ je isto što i $W^l$ jer $z^l = W^l a^{l-1} + b^l$
Kada ovo zamenimo desna strana postaje:
$$\frac{dC}{da^L} \odot (f^L)' \cdot W^L \odot (f^{L-1})' \cdot W^{L-1} \odot \dots \odot (f^1)' \cdot W^1$$
Pri čemu je $\odot$ hadamardov proizvod (elementwise mnozenje). Zasto ta operacija? Aktivaciona funkcija je elementwise operacija, pa za svaku aktivaciju u sloju l vazi: $$a_i^l = f(z_i^l)$$To znači da $a_1$ zavisi **samo** od $z_1$, $a_2$ samo od $z_2$, itd. Ako bismo napisali punu Jakobijanovu matricu $\frac{da^l}{dz^l}$ (tj prikazali zavisnost aktivacije svakog neurona po ulaznim sumama za sve neurone), ona bi bila **dijagonalna**$$\frac{da^l}{dz^l} = \begin{bmatrix} f'(z_1^l) & 0 & \dots \\ 0 & f'(z_2^l) & \dots \\ \vdots & \vdots & \ddots \end{bmatrix}$$Mnozenje matrica $A \cdot B$ gde je B dijagonalna matrica, isto je sto i $A \odot C$ gde je C vektor kolona sa elementima dijagonale matrice B.

U matematici, vektor gradient jednak je transponovanom total derivative vektoru. Tj gradient je vektor kolona a total derivative vektor red. 
Primenom pravila: $(A \cdot B)^T = B^T \cdot A^T$ dobija se gradijent cost funkcije: 
$$\nabla_x C = (W^1)^T \cdot (f^1)' \odot \dots \odot (W^{L-1})^T \cdot (f^{L-1})' \odot (W^L)^T \cdot (f^L)' \odot \nabla_{a^L} C$$
Sada mozemo definisati delta za poslednji sloj L:
$$ \delta^L = (f^L)' \odot \nabla_{a^L} C \tag{1}$$
Odatle sledi da je delta za bilo koji sloj $l$ jednaka:
$$\delta^l = ((W^{l+1})^T \delta^{l+1}) \odot f'(z^l) \tag{2}$$
Ovde se pojavljuje dosta matematike, ali u praksi koncpet je vrlo jednostavan.
___
## Azuriranje parametara
Znajuci osetljivost neurona na ulaznu sumu (delta), i znajuci tezine i aktivacije koje ucestvuju u toj ulaznoj sumi, mozemo promeniti te tezine tako da se **poprave srazmerno deltama i aktivacijama**. Sto su vece delte i aktivacije, to ce vise da smanjimo vrednost odgovarajuce tezine i time smanjimo vezu izmedju neurona j i k.
$$w_{jk}^l \leftarrow w_{jk}^l - \eta \cdot \frac{\partial C}{\partial w_{jk}^l}$$
gde je $\eta$ hiperparametar za skaliranje koraka, i vazi:
$$\frac{\partial C}{\partial w_{jk}^l} = \delta_j^l \cdot a_k^{l-1}$$
Što smo ranije pokazali.
___
Sada imamo 4 kljucne jednacine za backpropagation:
$$ \delta^L = (f^L)' \odot \nabla_{a^L} C$$
$$\delta^l = ((W^{l+1})^T \delta^{l+1}) \odot f'(z^l)$$
$$w_{jk}^l \leftarrow w_{jk}^l - \eta \cdot \delta_j^l \cdot a_k^{l-1}$$
$$\delta^l = \frac{\partial C}{\partial z^l}$$
## U ovom kodu
U ovom kodu parametri modela se azuriraju odmah za svaki datapoint:
1) Napravimo neuronsku mrezu i definisemo njene aktivacione funkcije.
2) Uradimo jedan forward pass za odredjeni datapoint (moze i batch) i sacuvamo aktivacije neurona i ulazne sume (tezinske sume).
3) U zavisnosti od izabrane cost funkcije, izracunamo gradijent cost funkcije koristeci dobijene aktivacije u zadnjem sloju i target. Potom koristimo f'(z^L) i izracunamo zadnji delta
4) za svaki sledeci (skriveni) sloj racunamo $\delta^l$ 
5) za taj sloj l azuriramo tezine po formuli
