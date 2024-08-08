```flow
start=>start: Start
end=>end: End

snr=>operation: Choose a SNR
bits=>operation: Generate random bits
mod=>operation: Modulation by Gray Code
awgn=>operation: Add to AWGN channel
demod=>operation: Demodulation to bits
err_cnt=>operation: Count error bits
err_100=>condition: If error bits > 100?
ber=>operation: Calculate BER
next_snr=>condition: Still need to calculate?

start(right)->snr->bits->mod->awgn->demod->err_cnt->err_100
err_100(no)->bits
err_100(yes)->ber->next_snr
next_snr(yes)->snr
next_snr(no)->end
```
