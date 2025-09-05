# Phase 3: Hierarchical Memory Implementation Plan

## Hedef
Mini DNC projesine hiyerarşik bellek sistemi entegre etmek.

## Mevcut Yapı
- Basic DNC model (basic_dnc.py)
- Simple memory bank (SimpleMemoryBank)
- Training infrastructure (train_basic_dnc.py)

## Yeni Hiyerarşik Bellek Yapısı

### 1. Multi-Level Memory Bank
```python
class HierarchicalMemoryBank(nn.Module):
    def __init__(self, d_model, slots=128, levels=3):
        super().__init__()
        self.d_model = d_model
        self.levels = levels
        
        # Her seviye için ayrı memory bank
        self.level_banks = nn.ModuleList()
        current_slots = slots
        for level in range(levels):
            bank = {
                'K': nn.Parameter(torch.randn(current_slots, d_model)),
                'V': nn.Parameter(torch.randn(current_slots, d_model)),
                'salience': nn.Parameter(torch.zeros(current_slots))
            }
            self.level_banks.append(nn.ParameterDict(bank))
            current_slots = current_slots // 2  # Her seviyede slot sayısını azalt
            
        # Compression modülleri
        self.compressors = nn.ModuleList([
            self._create_compressor(d_model, ratio=2**i) 
            for i in range(levels)
        ])
        
        # Routing ağı
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, levels),
            nn.Softmax(dim=-1)
        )
```

### 2. Compression Modülü
```python
class LearnedCompressor(nn.Module):
    def __init__(self, d_model, compression_ratio=2):
        super().__init__()
        self.compress_net = nn.Sequential(
            nn.Linear(d_model, d_model // compression_ratio),
            nn.LayerNorm(d_model // compression_ratio)
        )
        self.decompress_net = nn.Sequential(
            nn.Linear(d_model // compression_ratio, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, x, compress=True):
        if compress:
            return self.compress_net(x)
        else:
            return self.decompress_net(x)
```

### 3. Hierarchical Read/Write
```python
def read(self, q):
    """Hiyerarşik okuma işlemi"""
    B, T, D = q.shape
    route_weights = self.router(q)  # [B, T, levels]
    
    all_reads = []
    attention_maps = []
    
    for level_idx, (bank, compressor) in enumerate(zip(self.level_banks, self.compressors)):
        level_weight = route_weights[:, :, level_idx:level_idx+1]
        
        # Sorguyu sıkıştır (daha yüksek seviyeler için)
        if level_idx > 0:
            compressed_q = compressor(q, compress=True)
        else:
            compressed_q = q
            
        # Attention hesaplama
        K = bank['K']
        scores = torch.einsum('btd,sd->bts', compressed_q, K) / math.sqrt(D)
        scores = scores + bank['salience'].unsqueeze(0).unsqueeze(0)
        
        # Top-k selection
        topk_scores, topk_idx = torch.topk(scores, k=min(8, scores.shape[-1]), dim=-1)
        attention_weights = F.softmax(topk_scores, dim=-1)
        
        # Değerleri oku
        V = bank['V']
        V_selected = V[topk_idx]  # [B, T, k, D]
        read_vectors = torch.einsum('btk,btkd->btd', attention_weights, V_selected)
        
        # Routing ağırlığı uygula
        weighted_read = read_vectors * level_weight
        all_reads.append(weighted_read)
        attention_maps.append(attention_weights)
    
    # Tüm seviyelerden gelen okumaları birleştir
    final_read = torch.stack(all_reads, dim=0).sum(dim=0)
    return final_read, attention_maps

def write(self, h, attention_maps=None):
    """Hiyerarşik yazma işlemi"""
    if not self.training:
        return
        
    route_weights = self.router(h)
    
    for level_idx, (bank, compressor) in enumerate(zip(self.level_banks, self.compressors)):
        level_weight = route_weights[:, :, level_idx].mean()
        
        if level_weight > 0.05:  # Sadece önemli seviyelere yaz
            # Bilgiyi sıkıştır
            if level_idx > 0:
                compressed_h = compressor(h, compress=True)
                decompressed_h = compressor(compressed_h, compress=False)
            else:
                decompressed_h = h
                
            # En az önemli slotu güncelle
            pooled = decompressed_h.mean(dim=(0, 1))
            evict_idx = torch.argmin(bank['salience'])
            
            with torch.no_grad():
                bank['K'][evict_idx] = pooled.detach()
                bank['V'][evict_idx] = pooled.detach()
                bank['salience'][evict_idx] = 0.9 * bank['salience'][evict_idx] + 0.1
```

## Implementation Adımları

### 1. Hafta: Temel Hiyerarşik Bellek
- [ ] HierarchicalMemoryBank sınıfını oluştur
- [ ] Multi-level memory bank implementasyonu
- [ ] Basic routing mekanizması
- [ ] Simple read/write metodları

### 2. Hafta: Gelişmiş Özellikler
- [ ] Learned compression modüllerinin entegrasyonu
- [ ] Dynamic routing ve attention
- [ ] Salience tracking
- [ ] Memory visualization araçları

### 3. Hafta: Test ve Optimizasyon
- [ ] Unit testler
- [ ] Performance benchmark'lar
- [ ] Memory utilization analizi
- [ ] Training stability improvements

## Entegrasyon Adımları

1. **Yeni model sınıfı oluştur**:
   ```python
   class HierarchicalDNC(BasicDNC):
       def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=8, 
                    slots=128, levels=3, max_seq_len=512):
           super().__init__(vocab_size, d_model, n_layers, n_heads, slots, max_seq_len)
           # Hierarchical memory ile değiştir
           self.memory = HierarchicalMemoryBank(d_model, slots, levels)
   ```

2. **Training script güncellemesi**:
   - Yeni model tipini kullan
   - Memory utilization monitoring
   - Attention visualization

3. **API güncellemesi**:
   - Memory state endpoint
   - Attention map visualization
   - Performance metrics

## Beklenen Kazanımlar
1. **Bellek Kapasitesi**: Daha fazla bilgi saklama
2. **Uzun Sekans Performansı**: Daha iyi uzun sekans işleme
3. **Scalability**: Daha büyük modeller için uygun yapı
4. **Analiz Araçları**: Bellek kullanımını anlama ve optimize etme

Bu plan, mevcut mini DNC implementasyonuna hiyerarşik bellek ekleyerek Phase 3 hedeflerini gerçekleştirmeyi amaçlamaktadır.