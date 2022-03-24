# HyperSI
Hyperspectral Image toolbox

<ul>
    <li>
        A Imagem Hiperspectral (Hyperspectral Imaging, HSI) pode ser descrita como uma técnica que combina uma imagem 
        digital com a espectroscopia para se obter tanto as informações espaciais quanto espectrais de uma determinada amostra.
    </li>
    
    <li>
         HSI representa um conjunto de dados tridimensionais chamado hipercubo, com duas dimensões espaciais e uma 
         dimensão espectral. A  dimensão espectral da HSI é formada a partir da sobreposição de k imagens digitais, 
         onde cada imagem corresponde distribuição espacial da intensidade de um sinal em um determinado comprimento de onda λ.
    </li>
    
    <li>
        Quando combinadas as informações espaciais com as espectrais de uma amostra, se obtém uma estrutura tridimensional. 
        Que pode ser representada por I(x, y, λ). denominada como cubo hiperspectral. Cada pixel vai conter um espectro 
        contendo uma informação espectral da amostra analisada.
    </li>
    
    <li>       
        Para um determinado comprimento de onda λ, uma imagem pode ser vista utilizando uma escala de cinza ou uma outra 
        escala de cor para representar sua intensidade.
    </li>

</ul>
 
   
![Alt text](/Users/Usuario/pipeline_apresentacao/HyperSI/images/hsi.png?raw=true "HSI")
   
![Alt text](/Users/Usuario/pipeline_apresentacao/HyperSI/images/mat hsi.png?raw=true "HSI")
   

<ul>
    <li>
        A aquisição de HSI é realizada por um equipamento denominado câmera hiperespectral (Hyperspectral Camera, HSC). 
        Essa é adquirida capturando uma linha de cada vez, em que a amostra é disposta em uma bandeja que se movimenta 
        durante a digitalização
    </li>
    <li>
        A aquisição da HSI retorna três arquivos para calibração, sendo eles a referencia do Dark, a referencia do White, 
        e a referencia da própria amostra.  
    </li>
</ul>

   
![Alt text](/Users/Usuario/pipeline_apresentacao/HyperSI/images/camera hsi.png?raw=true "HSI")
   
   
<ul>
    <li style="text-align: justify">
        O processo da documentação inicia-se pelo notebook "classification_exemple" e as análises das classificações, 
        quando a Análise de Componentes Principais (PCA) e seus respectivos espectros no notebook "case". 
        Por fim, em "visualize_results" são plotadas as classificações pixel a pixel conforme imagem.
    </li>
</ul>
   