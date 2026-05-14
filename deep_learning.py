import io
import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error


def ibox(t): st.markdown(f'<div style="background:#0c1230;border-left:3px solid #6366f1;padding:12px 16px;border-radius:0 10px 10px 0;color:#a5b4fc;font-size:.84em;line-height:1.9;margin:8px 0;">💡 {t}</div>', unsafe_allow_html=True)
def sbox(t): st.markdown(f'<div style="background:#061a0f;border-left:3px solid #22c55e;padding:12px 16px;border-radius:0 10px 10px 0;color:#86efac;font-size:.84em;line-height:1.9;margin:8px 0;">✅ {t}</div>', unsafe_allow_html=True)
def wbox(t): st.markdown(f'<div style="background:#1a140a;border-left:3px solid #f59e0b;padding:12px 16px;border-radius:0 10px 10px 0;color:#fcd34d;font-size:.84em;line-height:1.9;margin:8px 0;">⚠️ {t}</div>', unsafe_allow_html=True)
def dbox(t): st.markdown(f'<div style="background:#1a0808;border-left:3px solid #ef4444;padding:12px 16px;border-radius:0 10px 10px 0;color:#fca5a5;font-size:.84em;line-height:1.9;margin:8px 0;">🔴 {t}</div>', unsafe_allow_html=True)
def divider(): st.markdown('<hr style="border:none;border-top:1px solid #1e293b;margin:16px 0;">', unsafe_allow_html=True)

def check_tensorflow():
    # Check if TensorFlow is available
    try:
        import tensorflow as tf
        return True, tf.__version__
    except ImportError:
        return False, None

def run_deep_learning(df):
    # Deep learning module UI
    tf_ok, tf_ver = check_tensorflow()

    st.markdown("""
    <div style='background:linear-gradient(145deg,#0f172a,#111827);border:1px solid #1e293b;
        border-radius:16px;padding:22px 26px;margin:10px 0;'>
    <div style='font-size:1.1em;font-weight:800;color:#f1f5f9;margin-bottom:4px;'>
        🧠 Deep Learning Models
    </div>
    <div style='color:#475569;font-size:.82em;'>
        Neural networks — fully connected, dropout, batch normalization, custom architectures
    </div>
    </div>
    """, unsafe_allow_html=True)

    if not tf_ok:
        dbox("TensorFlow not installed. Run: <code>pip install tensorflow</code>")
        st.markdown("""
        <div style='background:#0f172a;border:1px solid #1e293b;border-radius:12px;padding:20px;margin:12px 0;'>
        <div style='color:#a5b4fc;font-weight:700;margin-bottom:10px;'>📚 Deep Learning Concepts (Theory)</div>
        <div style='color:#cbd5e1;font-size:.83em;line-height:2;'>
        Even without TensorFlow, here is what these models do:<br><br>

        <b style='color:#f1f5f9;'>Neural Network Architecture:</b><br>
        Input Layer → Hidden Layers → Output Layer<br>
        Each neuron: z = Wx + b, activation: a = σ(z)<br><br>

        <b style='color:#f1f5f9;'>Activation Functions:</b><br>
        • ReLU: f(x) = max(0,x) — prevents vanishing gradient<br>
        • Sigmoid: f(x) = 1/(1+e⁻ˣ) — binary output<br>
        • Softmax: normalised exponentials — multiclass output<br>
        • Tanh: f(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) — zero-centred<br><br>

        <b style='color:#f1f5f9;'>Loss Functions:</b><br>
        • Binary Cross-Entropy — binary classification<br>
        • Categorical Cross-Entropy — multiclass<br>
        • MSE — regression<br>
        • Huber Loss — robust regression<br><br>

        <b style='color:#f1f5f9;'>Optimisers (gradient descent variants):</b><br>
        • SGD: θ = θ - α∇L — pure gradient descent<br>
        • Adam: Adaptive Moment Estimation — combines momentum + RMSProp<br>
        • RMSProp: Divides gradient by running average of magnitudes<br>
        • Adagrad: Adapts learning rate per parameter<br><br>

        <b style='color:#f1f5f9;'>Regularisation Techniques:</b><br>
        • Dropout: randomly deactivates neurons during training<br>
        • Batch Normalization: normalises layer inputs<br>
        • L1/L2: adds weight penalty to loss<br>
        • Early Stopping: stops when validation loss stops improving
        </div>
        </div>
        """, unsafe_allow_html=True)
        return

    import tensorflow as tf
    try:
        import keras
        from keras import layers, regularizers, callbacks
        from keras.utils import to_categorical
    except ImportError:
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers, callbacks
        from tensorflow.keras.utils import to_categorical
    sbox(f"TensorFlow {tf_ver} available")

    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include='number').columns.tolist()

    dl_tabs = st.tabs([
        "🏗️ Architecture Builder",
        "📚 Theory & Concepts",
        "⚡ Quick Templates"
    ])

    # ════════════════════════════════════════════════
    # ARCHITECTURE BUILDER
    # ════════════════════════════════════════════════
    with dl_tabs[0]:
        st.markdown("#### 🏗️ Custom Neural Network Builder")

        # Task setup
        t1,t2 = st.columns(2)
        with t1:
            dl_task   = st.selectbox("Task type", ["Binary Classification","Multi-class Classification","Regression"], key='dl_task')
        with t2:
            dl_target = st.selectbox("Target column", all_cols, key='dl_tgt')

        dl_feats = st.multiselect("Feature columns",
                                   [c for c in all_cols if c!=dl_target],
                                   default=[c for c in num_cols if c!=dl_target][:10],
                                   key='dl_feat')

        divider()
        st.markdown("#### 🔧 Network Architecture")

        a1,a2,a3 = st.columns(3)
        with a1: n_layers  = st.slider("Hidden layers", 1, 8, 3, key='n_lay')
        with a2: n_neurons = st.selectbox("Neurons per layer", [16,32,64,128,256,512,1024], index=3, key='n_neu')
        with a3: activation = st.selectbox("Activation function",
                                            ['relu','tanh','sigmoid','elu','selu','gelu'], key='activ')

        b1,b2,b3 = st.columns(3)
        with b1: dropout_rate = st.slider("Dropout rate", 0.0, 0.7, 0.2, 0.05, key='drop')
        with b2: use_batchnorm = st.checkbox("Batch Normalisation", value=True, key='bn')
        with b3: use_l2       = st.checkbox("L2 Regularisation",    value=False, key='l2r')

        divider()
        st.markdown("#### ⚙️ Training Configuration")

        c1,c2,c3,c4 = st.columns(4)
        with c1: optimizer  = st.selectbox("Optimiser",
                                            ['adam','sgd','rmsprop','adagrad','adamw'], key='optim')
        with c2: lr         = st.select_slider("Learning rate",
                                                [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1], value=0.001, key='dlr')
        with c3: epochs     = st.slider("Epochs", 10, 200, 50, key='ep')
        with c4: batch_size = st.selectbox("Batch size", [16,32,64,128,256,512], index=2, key='bs')

        d1,d2,d3 = st.columns(3)
        with d1: test_size_dl = st.slider("Test size %", 10, 40, 20, key='dl_ts')
        with d2: use_early_stop = st.checkbox("Early Stopping", value=True, key='es')
        with d3: patience = st.slider("ES Patience", 3, 20, 5, key='pat') if use_early_stop else 5

        val_split = st.slider("Validation split", 0.05, 0.3, 0.15, key='val_sp')

        # Architecture preview
        st.markdown("#### 📐 Architecture Preview")
        arch_lines = [f"Input Layer → ({len(dl_feats) if dl_feats else '?'} features)"]
        for i in range(n_layers):
            line = f"Dense({n_neurons}, activation='{activation}')"
            if use_batchnorm: line += " → BatchNorm"
            if dropout_rate > 0: line += f" → Dropout({dropout_rate})"
            arch_lines.append(f"Hidden Layer {i+1}: {line}")

        if dl_task == "Binary Classification":
            arch_lines.append("Output: Dense(1, activation='sigmoid') → Binary output")
        elif dl_task == "Multi-class Classification":
            arch_lines.append("Output: Dense(n_classes, activation='softmax') → Class probabilities")
        else:
            arch_lines.append("Output: Dense(1, activation='linear') → Continuous value")

        st.markdown(f"""
        <div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;
            padding:16px;font-family:monospace;font-size:.8em;'>
        {'<br>'.join([f"<div style='color:#a5b4fc;padding:3px 0;'>→ {l}</div>" for l in arch_lines])}
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Train Neural Network", key='train_dl'):
            if not dl_feats:
                wbox("Select feature columns.")
                st.stop()

            # Prepare data
            ml_df = df[dl_feats + [dl_target]].copy().dropna()
            if len(ml_df) < 20:
                dbox("Need at least 20 rows.")
                st.stop()

            le = LabelEncoder()
            for col in ml_df.select_dtypes(include='object').columns:
                ml_df[col] = le.fit_transform(ml_df[col].astype(str))

            X = np.asarray(ml_df[dl_feats], dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            y_raw = np.asarray(ml_df[dl_target], dtype=object)

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            # Encode target
            if dl_task == "Binary Classification":
                if y_raw.dtype == object:
                    y_raw = le.fit_transform(y_raw.astype(str))
                y = y_raw.astype(np.float32)
                output_units   = 1
                output_act     = 'sigmoid'
                loss_fn        = 'binary_crossentropy'
                metric_name    = 'accuracy'

            elif dl_task == "Multi-class Classification":
                if y_raw.dtype == object:
                    y_raw = le.fit_transform(y_raw.astype(str))
                n_cls = len(np.unique(y_raw))
                y     = to_categorical(y_raw, num_classes=n_cls)
                output_units = n_cls
                output_act   = 'softmax'
                loss_fn      = 'categorical_crossentropy'
                metric_name  = 'accuracy'

            else:  # Regression
                y = y_raw.astype(np.float32)
                output_units = 1
                output_act   = 'linear'
                loss_fn      = 'mse'
                metric_name  = 'mae'

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_dl/100, random_state=42
            )

            # Build model
            l2_reg = regularizers.l2(0.001) if use_l2 else None

            model = keras.Sequential()
            model.add(layers.Input(shape=(X_train.shape[1],)))

            for i in range(n_layers):
                model.add(layers.Dense(n_neurons, activation=activation,
                                        kernel_regularizer=l2_reg))
                if use_batchnorm:
                    model.add(layers.BatchNormalization())
                if dropout_rate > 0:
                    model.add(layers.Dropout(dropout_rate))

            model.add(layers.Dense(output_units, activation=output_act))

            # Compile
            opt_map = {
                'adam':    keras.optimizers.Adam(learning_rate=lr),
                'sgd':     keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
                'rmsprop': keras.optimizers.RMSprop(learning_rate=lr),
                'adagrad': keras.optimizers.Adagrad(learning_rate=lr),
                'adamw':   keras.optimizers.AdamW(learning_rate=lr),
            }
            model.compile(
                optimizer=opt_map.get(optimizer, 'adam'),
                loss=loss_fn,
                metrics=[metric_name]
            )

            # Model summary
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            with st.expander("📋 Model Summary"):
                st.code('\n'.join(summary_lines))

            total_params = model.count_params()
            st.metric("Total Parameters", f"{total_params:,}")

            # Callbacks
            cb_list = []
            if use_early_stop:
                cb_list.append(callbacks.EarlyStopping(
                    monitor='val_loss', patience=patience,
                    restore_best_weights=True, verbose=0
                ))
            cb_list.append(callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=max(2,patience//2), verbose=0
            ))

            # Train
            progress_bar = st.progress(0)
            status_text  = st.empty()

            class StreamlitCallback(keras.callbacks.Callback):
                def __init__(self):
                    self.epoch_losses   = []
                    self.epoch_val_loss = []

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    self.epoch_losses.append(logs.get('loss',0))
                    self.epoch_val_loss.append(logs.get('val_loss',0))
                    pct = int((epoch+1)/epochs*100)
                    progress_bar.progress(min(pct,100))
                    status_text.markdown(
                        f"Epoch {epoch+1}/{epochs} — "
                        f"Loss: {logs.get('loss',0):.4f} — "
                        f"Val Loss: {logs.get('val_loss',0):.4f}"
                    )

            sl_cb = StreamlitCallback()
            cb_list.append(sl_cb)

            with st.spinner("Training neural network..."):
                history = model.fit(
                    X_train, y_train,
                    validation_split=val_split,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=cb_list,
                    verbose=0
                )

            progress_bar.progress(100)
            status_text.empty()

            # Training curves
            st.markdown("#### 📈 Training History")
            hist_df = pd.DataFrame({
                'Epoch':    range(1, len(history.history['loss'])+1),
                'Train Loss':      history.history['loss'],
                'Val Loss':        history.history.get('val_loss',[]),
            })
            st.dataframe(hist_df.tail(10).round(4), use_container_width=True)

            final_loss    = history.history['loss'][-1]
            final_val_loss = history.history.get('val_loss',['N/A'])[-1]
            epochs_ran    = len(history.history['loss'])

            e1,e2,e3 = st.columns(3)
            e1.metric("Final Train Loss",  f"{final_loss:.4f}")
            e2.metric("Final Val Loss",    f"{final_val_loss:.4f}" if isinstance(final_val_loss, float) else final_val_loss)
            e3.metric("Epochs Completed",  epochs_ran)

            # Evaluate
            divider()
            st.markdown("#### 🎯 Test Set Evaluation")

            if dl_task == "Binary Classification":
                y_prob = model.predict(X_test, verbose=0).flatten()
                y_pred = (y_prob >= 0.5).astype(int)
                acc = accuracy_score(y_test, y_pred)
                st.metric("Test Accuracy", f"{acc*100:.2f}%")
                from sklearn.metrics import classification_report
                try:
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
                except: pass

            elif dl_task == "Multi-class Classification":
                y_prob = model.predict(X_test, verbose=0)
                y_pred = np.argmax(y_prob, axis=1)
                y_true = np.argmax(y_test, axis=1)
                acc = accuracy_score(y_true, y_pred)
                st.metric("Test Accuracy", f"{acc*100:.2f}%")

            else:  # Regression
                y_pred = model.predict(X_test, verbose=0).flatten()
                r2   = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r1,r2_ = st.columns(2)
                r1.metric("R² Score", f"{r2:.4f}")
                r2_.metric("RMSE",    f"{rmse:.4f}")

            sbox(f"Neural network trained. {epochs_ran} epochs completed. {total_params:,} parameters.")

    # ════════════════════════════════════════════════
    # THEORY TAB
    # ════════════════════════════════════════════════
    with dl_tabs[1]:
        st.markdown("""
        <div style='background:#0f172a;border:1px solid #1e293b;border-radius:14px;padding:24px;'>
        <div style='color:#f1f5f9;font-size:1em;font-weight:800;margin-bottom:16px;'>
            📚 Deep Learning — Complete Theory Reference
        </div>
        <div style='color:#cbd5e1;font-size:.83em;line-height:2.2;'>

        <b style='color:#a5b4fc;font-size:.9em;'>🧠 FORWARD PROPAGATION</b><br>
        z¹ = W¹·X + b¹ &nbsp;|&nbsp; a¹ = σ(z¹)<br>
        z² = W²·a¹ + b² &nbsp;|&nbsp; a² = σ(z²)<br>
        Output = aᴸ where L = number of layers<br><br>

        <b style='color:#a5b4fc;font-size:.9em;'>⬅️ BACKPROPAGATION</b><br>
        δᴸ = ∂L/∂aᴸ · σ'(zᴸ) &nbsp; (output layer error)<br>
        δˡ = (Wˡ⁺¹)ᵀ·δˡ⁺¹ · σ'(zˡ) &nbsp; (hidden layer error)<br>
        ∂L/∂Wˡ = δˡ·(aˡ⁻¹)ᵀ &nbsp; (weight gradient)<br>
        ∂L/∂bˡ = δˡ &nbsp; (bias gradient)<br><br>

        <b style='color:#a5b4fc;font-size:.9em;'>📉 LOSS FUNCTIONS</b><br>
        Binary CE: L = -[y·log(p) + (1-y)·log(1-p)]<br>
        Categorical CE: L = -Σ yᵢ·log(pᵢ)<br>
        MSE: L = (1/n)·Σ(y-ŷ)² &nbsp;|&nbsp; MAE: L = (1/n)·Σ|y-ŷ|<br>
        Huber: L = MSE if |e|≤δ, MAE·δ otherwise<br><br>

        <b style='color:#a5b4fc;font-size:.9em;'>⚡ OPTIMISERS</b><br>
        SGD: θ = θ - α·∇L<br>
        Momentum: v = β·v - α·∇L, θ = θ + v<br>
        RMSProp: θ = θ - α·∇L/√(E[g²]+ε)<br>
        Adam: combines momentum + RMSProp → most popular<br>
        AdamW: Adam + weight decay (better generalisation)<br><br>

        <b style='color:#a5b4fc;font-size:.9em;'>🔧 REGULARISATION</b><br>
        Dropout: randomly zeroes neurons during training → prevents co-adaptation<br>
        Batch Norm: normalises layer input → faster training, less sensitive to init<br>
        L1: λ·Σ|w| → sparsity &nbsp;|&nbsp; L2: λ·Σw² → weight shrinkage<br>
        Early Stopping: stops when val_loss stops improving<br><br>

        <b style='color:#a5b4fc;font-size:.9em;'>📊 METRICS</b><br>
        Classification: Accuracy, Precision, Recall, F1, ROC-AUC<br>
        Regression: R², RMSE, MAE, MAPE<br>
        Convergence: Loss curve should decrease smoothly — noisy = high LR, flat = low LR
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════
    # QUICK TEMPLATES
    # ════════════════════════════════════════════════
    with dl_tabs[2]:
        st.markdown("#### ⚡ Quick Architecture Templates")
        ibox("Pre-built architectures for common use cases. Select and customise.")

        templates = {
            "🏠 Simple MLP (small data)":
                "2 hidden layers × 64 neurons, ReLU, Dropout 0.2, Adam 0.001",
            "🏢 Deep MLP (large data)":
                "5 hidden layers × 256 neurons, ReLU, BatchNorm + Dropout 0.3, AdamW 0.0005",
            "📊 Tabular Specialist":
                "3 layers × 128 neurons, GELU activation, L2 reg, BatchNorm, Adam 0.001",
            "⚡ Fast Prototype":
                "1 hidden layer × 32 neurons, ReLU, SGD 0.01 (quick experiments)",
            "🎯 High Accuracy":
                "6 layers × 512 neurons, ELU, BatchNorm, Dropout 0.4, AdamW 0.0003, 200 epochs",
        }

        for name, desc in templates.items():
            st.markdown(f"""
            <div style='background:#0f172a;border:1px solid #1e293b;border-radius:10px;
                padding:14px 18px;margin:6px 0;display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <div style='color:#f1f5f9;font-weight:700;font-size:.88em;'>{name}</div>
                    <div style='color:#475569;font-size:.78em;margin-top:4px;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)